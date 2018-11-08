from functools import reduce
from operator import mul

import cplex
import numpy as np
import tensorflow as tf

from RL.common import logger


def convert_to_constraints_dict(nbases, nresources, min_constraints, max_constraints):
    constraints = {
        "name": "root_node",
        "equals": nresources,
        "max": nresources,
        "min": nresources,
        "children": []
    }
    for i in range(nbases):
        child = {
            "name": "zone{0}".format(i),
            "zone_id": i,
            "equals": None,
            "min": min_constraints[i],
            "max": max_constraints[i],
            "children": []
        }
        constraints["children"].append(child)
    return constraints


def normalize_constraints(constraints, nresources=None):
    if constraints['equals'] is not None:
        nresources = constraints['equals']
        constraints['equals'] = constraints['equals'] / nresources
    assert nresources is not None, "Number of resources not specified"
    constraints['min'] = constraints['min'] / nresources
    constraints['max'] = constraints['max'] / nresources
    if 'children' in constraints:
        for child_constraints in constraints['children']:
            normalize_constraints(child_constraints, nresources)


def depth_of_constraints(constraints):
    children = constraints.get('children', [])
    if len(children) == 0:
        return 0
    else:
        return 1 + max(depth_of_constraints(child) for child in children)


def count_leaf_nodes_in_constraints(constraints):
    if 'children' not in constraints or len(constraints['children']) == 0:
        return 1
    else:
        count = 0
        for child_constraints in constraints['children']:
            count += count_leaf_nodes_in_constraints(child_constraints)
        return count


def count_nodes_in_constraints(constraints):
    count = 1
    if 'children' in constraints:
        for child_constraints in constraints['children']:
            count += count_nodes_in_constraints(child_constraints)
    return count


def get_zone_ids_under(constraints):
    children = constraints.get('children', [])
    if len(children) == 0:
        ids = [constraints['zone_id']]
    else:
        ids = []
        for c in children:
            c_ids = get_zone_ids_under(c)
            ids.extend(c_ids)
    return ids


def tf_safe_softmax(inputs, scope):
    with tf.variable_scope(scope):
        x = inputs - tf.reduce_max(inputs, axis=1, keepdims=True)
        # exp = tf.exp(tf.minimum(inputs, 0))
        exp = tf.exp(x)
        sigma = tf.reduce_sum(exp, axis=-1, keepdims=True, name='sum')
        return exp / sigma
    # return tf_safe_softmax_with_non_uniform_individual_constraints(inputs, [0.1875] * 25, scope)


def tf_safe_softmax_with_non_uniform_individual_constraints(inputs, constraints, scope):
    """adds a max_constrained_softmax layer to compution graph, with as many outputs as inputs

    Arguments:
        inputs {tensor} -- raw (unbounded) output of neural network
        constraints {numpy.ndarray} -- array of max constraints, should of same shape as inputs.
                                        Should sum to more than 1. Each constraint should be in (0, 1].
        scope {string} -- tensorflow name scope for the layer

    Raises:
        ValueError -- if constraints are invalid.

    Returns:
        [tensor] -- s.t. sum of outputs = 1. each output in (0, 1). each output <= corresponding constraint.
    """

    with tf.variable_scope(scope):
        inputs_shape = inputs.shape.as_list()[1:]
        dimensions = reduce(mul, inputs_shape, 1)
        constraints = np.asarray(constraints)
        if list(constraints.shape) != inputs_shape:
            raise ValueError('shape of constraints {0} not compatible with shape of inputs {1}'.format(
                constraints.shape, inputs_shape))
        if np.any(constraints <= 0) or np.any(constraints > 1):
            raise ValueError(
                "constraints need to be in range (0, 1]")
        if np.sum(constraints) <= 1:
            raise ValueError(
                "sum of max constraints needs to be greater than 1")

        # x = inputs - tf.reduce_max(inputs, axis=1, keepdims=True)
        x = tf.minimum(inputs, 0)
        y = tf.exp(x)
        sigma = tf.reduce_sum(y, axis=-1, keepdims=True, name='sum')

        '''
        for some epsilons vector,
        our output z needs to be (y + epsilons)/(sigma + sum(epsilons))
        to satisfy the constraints, we get the following set of linear equations:
        for all i:
            (constraints[i] - 1) * epsilons[i] + constraints[i] * sum(epsilons[except i]) = 1 - constraints[i]
        '''
        constraints_flat = constraints.flatten()
        # to solve the epsilons linear equations: coeffs * epsilons = constants
        # coefficient matrix:
        coeffs = np.array([[(constraints_flat[row] - 1 if col == row else constraints_flat[row])
                            for col in range(dimensions)] for row in range(dimensions)])
        constants = np.array([1 - constraints_flat[row]
                              for row in range(dimensions)])
        epsilons_flat = np.linalg.solve(coeffs, constants)
        epsilons = np.reshape(epsilons_flat, inputs_shape)
        logger.log("constrained_softmax_max: episilons are {0}".format(
            epsilons), level=logger.INFO)
        assert np.all(epsilons >= 0), "The given constraints are not supported yet. They should satisfy for all k, C_k >= (sum(C) - 1)/(len(C)-1)"
        epsilons_sigma = np.sum(epsilons)
        return (y + epsilons) / (sigma + epsilons_sigma)


def tf_max_constrained_softmax(inputs, max_constraints, scope):
    return tf_safe_softmax_with_non_uniform_individual_constraints(inputs, max_constraints, scope)


def tf_max_constrained_softmax_with_dynamic_epsilons(inputs, constraints, scope):
    with tf.variable_scope(scope):
        inputs_shape = inputs.shape.as_list()[1:]
        dimensions = reduce(mul, inputs_shape, 1)

        x = tf.minimum(inputs, 0)
        y = tf.exp(x)
        sigma = tf.reduce_sum(y, axis=-1, keepdims=True, name='sum')

        '''
        for some epsilons vector,
        our output z needs to be (y + epsilons)/(sigma + sum(epsilons))
        to satisfy the constraints, we get the following set of linear equations:
        for all i:
            (constraints[i] - 1) * epsilons[i] + constraints[i] * sum(epsilons[except i]) = 1 - constraints[i]
        '''
        constraints_flat = tf.reshape(
            constraints, [-1, dimensions], name='constraints_1D')
        # to solve the epsilons linear equations: coeffs * epsilons = constants
        # coefficient matrix:
        coeffs = tf.convert_to_tensor([[(constraints_flat[:, row] - 1 if col == row else constraints_flat[:, row])
                                        for col in range(dimensions)] for row in range(dimensions)], name='coeffs')
        coeffs = tf.transpose(coeffs, [2, 0, 1], name='coeffs_2D')
        constants = tf.convert_to_tensor([1 - constraints_flat[:, row]
                                          for row in range(dimensions)], name='constants')
        constants_1D = tf.transpose(constants, [1, 0], name='constants_1D')
        constants_2D = tf.expand_dims(
            constants_1D, axis=-1, name='constants_2D')
        # epsilons_flat = np.linalg.solve(coeffs, constants)
        coeffs_inverse = tf.matrix_inverse(coeffs, name='coeffs_inverse')
        epsilons = tf.reshape(tf.matmul(coeffs_inverse, constants_2D,
                                        name='epsilons_2D'), [-1, dimensions], name='epsilons_1D')
        # epsilons = np.reshape(epsilons_flat, inputs_shape)
        # logger.log("constrained_softmax__dynamic_max: episilons are {0}".format(
        #     epsilons), level=logger.INFO)
        epsilons_sigma = tf.reduce_sum(
            epsilons, axis=-1, keepdims=True, name='epsilons_sum')
        # epsilons = tf.Print(
        #     epsilons, [epsilons], "constrained_softmax_dynamic_epsilons: ")
        print("epsilons shape", epsilons.shape.as_list())
        return (y + epsilons) / (sigma + epsilons_sigma)  # [batch, dimensions]


def tf_minmax_constrained_softmax(inputs, min_constraints, max_constraints, scope):
    '''we want z_i to sum to 1. s.t. z_i in [m_i, M_i].\n
    so we distribute m_i to z_i first.\n
    then the problem statement becomes:\n
    find vector u s.t. u_i sums to s=1-sum(m_i) and u_i in [0, M_i - m_i]\n
    to do that, we do u = s * max_constrained_softmax(inputs, (M-m)/s)\n
    then z_i = m_i + u_i

    Arguments:
        inputs {tensor} -- raw (unbounded) output of neural network
        min_constraints {numpy.ndarray} -- of same shape as inputs. each component in [0,1). should sum to < 1
        max_constraints {numpy.ndarray} -- of same shape as inputs. each component in (0,1]. should sum to > 1
        scope {str} -- tensorflow name scope for the layer
    '''
    inputs_shape = inputs.shape.as_list()[1:]
    if list(max_constraints.shape) != inputs_shape:
        raise ValueError('shape of max_constraints {0} is not compatible with shape of inputs {1}'.format(
            max_constraints.shape, inputs_shape))
    if list(min_constraints.shape) != inputs_shape:
        raise ValueError('shape of min_constraints {0} is not compatible with shape of inputs {1}'.format(
            min_constraints.shape, inputs_shape))
    if np.any(max_constraints <= 0) or np.any(max_constraints > 1):
        raise ValueError(
            "max_constraints need to be in range (0, 1]")
    if np.any(min_constraints < 0) or np.any(min_constraints >= 1):
        raise ValueError(
            "min_constraints need to be in range [0, 1)")
    if np.any(max_constraints <= min_constraints):
        raise ValueError(
            "max_constraints need to strictly greater than min_constraints")
    if np.sum(max_constraints) <= 1:
        raise ValueError("sum of max_constraints needs to be greater than 1")
    if np.sum(min_constraints) >= 1:
        raise ValueError("sum of min_constraints needs to be less than 1")

    with tf.variable_scope(scope):
        s = 1 - np.sum(min_constraints)
        u_max_constraints = np.minimum(
            1.0, (max_constraints - min_constraints) / s)
        u = s * tf_max_constrained_softmax(
            inputs, u_max_constraints, "max_constrained_softmax")  # [batch, dimensions]
        z = min_constraints + u
        return z


def _tf_nested_constrained_softmax(inputs, constrained_node, scope, z_tree={}, z_node=None):
    '''This algo distributes the value of z_node["tensor"] among its children.
    When called with root_node, assumes that total number of inputs = number of leaf nodes in the constraints_tree\n
    Should be initially called with root_node of constraints tree and empty z_tree and null z_node.
    '''
    with tf.variable_scope(scope):
        # root node case:
        if constrained_node["equals"] is not None:
            assert len(z_tree.keys(
            )) == 0, "An equals_constraint was encountered at a non-root node! Node name: {0}".format(constrained_node["name"])
            assert constrained_node["equals"] == constrained_node["min"] == constrained_node[
                "max"], "At root node, min, max & equal constraints should be same"
            # initialize z_tree:
            z_tree["tensor"] = constrained_node["equals"]
            z_tree["equals"] = constrained_node["equals"]
            z_tree["min"] = constrained_node["min"]
            z_tree["max"] = constrained_node["max"]
            z_tree["name"] = constrained_node["name"]
            z_tree["consumed_inputs"] = 0
            z_node = z_tree

        # base_case: if this the leaf node, return the tensor
        if "children" not in constrained_node or len(constrained_node["children"]) == 0:
            z_node["zone_id"] = constrained_node["zone_id"]
            return [z_node]

        '''
        M = max_constraints of children\n
        m = min_constraints of children\n
        We will distribute min_constraints to the respective nodes first. \n
        The remaining sum will be: \n
            s = z_node["tensor"] - sum(m) \n
        The maximum value of this sum can be:
            S = z_node["max"] - sum(m) \n
        Then we find vector u s.t. sum(u)=s. and u_i in (0, M_i-m_i) \n
            u = s * max_constrained_softmax(inputs, (M-m)/S) \n
        Then: \n
            z_children_i = m_i + u_i \n
        i.e. \n
            z_children_i = m_i + s * max_constrained_softmax(inputs, (M - m)/S)_i \n

        Please convince yourself that z_children_i will be in range (m_i, M_i) and sum(z_children) = z_node["tensor"]
        '''

        children = constrained_node["children"]
        max_constraints = np.array([c["max"] for c in children])
        min_constraints = np.array([c["min"] for c in children])
        assert 0 <= np.sum(min_constraints) <= z_node["min"] <= z_node["max"] < np.sum(
            max_constraints), "The constraints should satisfy 0 <= sum(children_min) <= parent_min <= parent_max < sum(children_max)"
        assert np.all(0 <= min_constraints) and np.all(min_constraints < max_constraints) and np.all(
            max_constraints <= 1), "The constraints should satisfy 0 <= min_constraint_i < max_constraint_i <= 1"

        # print(constrained_node['name'])
        if constrained_node['equals'] is not None:
            # then this is root node. we need not compute epsilons dynamically
            s = z_node["tensor"] - np.sum(min_constraints)
            u_max_constraints = (max_constraints - min_constraints) / s
            u_max_constraints = np.minimum(u_max_constraints, 1)
            u = s * tf_max_constrained_softmax(inputs[:, z_tree["consumed_inputs"]:z_tree["consumed_inputs"] + len(
                children)], u_max_constraints, scope="MCSM_on_children_of_{0}".format(z_node["name"]))
            z_tree["consumed_inputs"] += len(children)
            z_children_tensors = min_constraints + u
        else:
            min_constraints_tensor = tf.constant(min_constraints, dtype=tf.float32, shape=[
                1, len(children)], name='min_constraints')
            max_constraints_tensor = tf.constant(max_constraints, dtype=tf.float32, shape=[
                1, len(children)], name='max_constraints')
            s = z_node["tensor"] - tf.reduce_sum(min_constraints_tensor)
            # S = z_node["max"] - sum(min_constraints)
            u_max_constraints = (max_constraints_tensor -
                                 min_constraints_tensor) / s
            u_max_constraints = tf.minimum(
                u_max_constraints, tf.constant(1.0, dtype=tf.float32))
            u = s * tf_max_constrained_softmax_with_dynamic_epsilons(inputs[:, z_tree["consumed_inputs"]:z_tree["consumed_inputs"] + len(
                children)], u_max_constraints, scope="MCSMD_on_children_of_{0}".format(z_node["name"]))
            z_tree["consumed_inputs"] += len(children)
            z_children_tensors = min_constraints_tensor + u

        z_node["children"] = []
        for i in range(len(children)):
            child = {}
            z_node["children"].append(child)
            child["tensor"] = z_children_tensors[:, i:i + 1]
            child["name"] = children[i]["name"]
            child["min"] = children[i]["min"]
            child["max"] = children[i]["max"]
            child["equals"] = children[i]["equals"]

        return_nodes = []
        for z_node_child, constrained_node_child in zip(z_node["children"], constrained_node["children"]):
            c_returned_nodes = _tf_nested_constrained_softmax(
                inputs, constrained_node_child, z_tree=z_tree, z_node=z_node_child, scope="NCSM_on_{0}".format(z_node_child["name"]))
            return_nodes.extend(c_returned_nodes)

        return return_nodes


def tf_nested_constrained_softmax(inputs, constraints, scope, z_tree={}):
    with tf.variable_scope(scope):
        leaf_z_nodes = _tf_nested_constrained_softmax(
            inputs, constraints, "_" + scope, z_tree=z_tree)
        leaf_z_tensors = [leaf['tensor'] for leaf in sorted(
            leaf_z_nodes, key=lambda leaf:leaf["zone_id"])]
        return tf.concat(values=leaf_z_tensors, axis=-1, name='concat_leaf_tensors')


def tf_infeasibility(actions, constraints_node, scope):
    with tf.variable_scope(scope):
        with tf.variable_scope('sum'):
            _sum = 0
            zone_ids = get_zone_ids_under(constraints_node)
            for zone_id in zone_ids:
                _sum += actions[:, zone_id]  # shape = [batch_size]
        with tf.variable_scope('sum_violation'):
            if constraints_node.get('equals', None) is not None:
                sum_violation = tf.abs(_sum - constraints_node['equals'])
            else:
                sum_violation = 0
        with tf.variable_scope('min_violation'):
            min_violation = tf.maximum(0.0, constraints_node["min"] - _sum)
        with tf.variable_scope('max_violation'):
            max_violation = tf.maximum(0.0, _sum - constraints_node["max"])

        for c in constraints_node.get('children', []):
            c_sum_violtion, c_min_violation, c_max_violation = tf_infeasibility(
                actions, c, 'infeasibility_{0}'.format(c['name']))
            sum_violation += c_sum_violtion
            min_violation += c_min_violation
            max_violation += c_max_violation

        return sum_violation, min_violation, max_violation


def _get_lp_rows(constraints_node):
    rows = []
    rhs = []
    names = []
    senses = []

    zone_ids = get_zone_ids_under(constraints_node)
    if constraints_node.get('equals', None) is not None:
        equals_row = [['z{0}'.format(id) for id in zone_ids], [
            1] * len(zone_ids)]
        rows.append(equals_row)
        rhs.append(constraints_node['equals'])
        senses.append('E')
        names.append('{0}_equals'.format(constraints_node['name']))

    min_row = [['z{0}'.format(id) for id in zone_ids], [1] * len(zone_ids)]
    rows.append(min_row)
    rhs.append(constraints_node['min'])
    senses.append('G')
    names.append('{0}_min'.format(constraints_node['name']))

    max_row = [['z{0}'.format(id) for id in zone_ids], [1] * len(zone_ids)]
    rows.append(max_row)
    rhs.append(constraints_node['max'])
    senses.append('L')
    names.append('{0}_max'.format(constraints_node['name']))

    for c in constraints_node.get('children', []):
        c_rows, c_rhs, c_senses, c_names = _get_lp_rows(c)
        rows.extend(c_rows)
        rhs.extend(c_rhs)
        senses.extend(c_senses)
        names.extend(c_names)

    return rows, rhs, senses, names


def cplex_nearest_feasible(action, constraints):
    '''
    let d_i = |z_i - a_i|
    obj = minimize d_1 + d_2 ... d_k
    subject to:
        bounds:
            d_i in [0, inf]
            z_i in [0, 1]
        constraints:
            for all i:
                d_i >= z_i - a_i
                d_i >= -(z_i - a_i)
            and:
                nested constraints
    '''
    action = list(action)
    k = len(action)
    # these are my output variables
    z_names = ["z{0}".format(i) for i in range(k)]
    # these are my difference variables. d_i = |z_i - a_i|
    d_names = ["d{0}".format(i) for i in range(k)]

    prob = cplex.Cplex()

    # objective and bounds set:
    prob.objective.set_sense(
        prob.objective.sense.minimize)  # we want to minimize
    names = d_names + z_names
    obj = [1] * k + [0] * k
    lb = [0] * k + [0] * k
    ub = [cplex.infinity] * k + [1] * k
    prob.variables.add(obj=obj, lb=lb, ub=ub, names=names)

    # d constraints:
    # d_i - z_i >= -a_i
    # d_i + z_i >= a_i
    rows = []
    names = []
    rhs = []
    for i in range(k):
        row_dgza = [["d{0}".format(i), "z{0}".format(i)], [1, -1]]
        row_dgaz = [["d{0}".format(i), "z{0}".format(i)], [1, 1]]
        rows.append(row_dgza)
        rows.append(row_dgaz)
        rhs.append(-action[i])
        rhs.append(action[i])
        names.append('d{0}_gza'.format(i))
        names.append('d{0}_gaz'.format(i))
    assert len(rows) == len(names) == len(rhs) == 2 * \
        k, print(len(rows), len(names), len(rhs), 2 * k)
    prob.linear_constraints.add(
        lin_expr=rows, senses='G' * (2 * k), rhs=rhs, names=names)

    # constraints:
    rows, rhs, senses, names = _get_lp_rows(constraints)
    senses = ''.join(senses)
    prob.linear_constraints.add(
        lin_expr=rows, senses=senses, rhs=rhs, names=names)

    prob.set_results_stream(None)
    prob.set_log_stream(None)

    prob.solve()

    # prob.write("nnmM.lp")
    feasible_action = np.array(prob.solution.get_values()[k:])
    feasible_action = np.clip(feasible_action, 0, 1)
    return {
        'feasible_action': feasible_action,
        'modification': feasible_action - np.array(action),
        'status': prob.solution.status[prob.solution.get_status()],
        'L1_diff': prob.solution.get_objective_value(),
        'prob': prob
    }
