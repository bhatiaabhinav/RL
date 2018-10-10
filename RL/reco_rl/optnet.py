from RL.reco_rl.constraints import cplex_nearest_feasible


class OptnetLayer:
    def __init__(self, n_outputs, n_inputs):
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs

    def forward_pass(self, input_vector):
        raise NotImplementedError()

    def gradients(self):
        '''
        returns gradients (Jacobian matrix) of the current output w.r.t current input
        by the method of differentiating the KKT conditions at the optimal point w.r.t the input variables
        '''
        raise NotImplementedError()


class ConstrainedProjectionOptnetLayer(OptnetLayer):
    def __init__(self, n_outputs, n_inputs, constraints):
        super().__init__(n_outputs, n_inputs)
        self.constraints = constraints

    def forward_pass(self, input_vector):
        self._current_input = input_vector
        output = cplex_nearest_feasible(input_vector, self.constraints)[
            'feasible_action']
        return output

    def gradients(self):
        '''
        returns gradients (Jacobian matrix) of the current output w.r.t current input
        by the method of differentiating the KKT conditions of the LP at the optimal point w.r.t the input variables
        '''
        raise NotImplementedError()
