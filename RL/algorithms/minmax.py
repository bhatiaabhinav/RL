from RL.envs.tictactoe import TicTacToe
import RL
from RL.agents import SeedingAgent, EnvRenderingAgent, MinMaxNetsAgent, ModelLoaderSaverAgent, RandomPlayAgent, PlayerTurnCoordinationAgent, PygletLoopAgent, SlowHuman  # noqa: F401


class MyContext(RL.Context):
    def make_env(self, env_id):
        if 'TicTacToe' in env_id:
            board_size = int(env_id.split('-')[1])
            return TicTacToe(board_size=board_size)
        else:
            return super().make_env(env_id)


my_context = MyContext()
runner = RL.Runner(my_context)

runner.register_agent(SeedingAgent(my_context, "seeder"))
runner.register_agent(EnvRenderingAgent(my_context, "renderer", auto_dispatch_on_render=False, episode_interval=my_context.render_interval))
# p1 = r.register_agent(RandomPlayAgent(c, "random_player"))
# p1 = runner.register_agent(MinMaxAgent(my_context, "cpu_minmax"))
p1 = runner.register_agent(MinMaxNetsAgent(my_context, "cpu_minmaxnets", "{0}_Model".format(my_context.env_id), max_depth=my_context.max_depth))
runner.register_agent(ModelLoaderSaverAgent(my_context, "model_loader_saver", p1.model.params))
# p2 = runner.register_agent(SlowHuman(my_context, "human_player"))
p2 = runner.register_agent(RandomPlayAgent(my_context, "random_player"))
runner.register_agent(PlayerTurnCoordinationAgent(my_context, "coordinator", [p1, p2], shuffle=False))
runner.register_agent(PygletLoopAgent(my_context, "pyglet_loop"))

runner.run()
