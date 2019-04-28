import RL
import os
from RL.common.utils import TFParamsSaverLoader


class ModelLoaderSaverAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, params, load_dir=None, save_sub_dir='models', filename='model'):
        super().__init__(context, name)
        self.loader_saver = TFParamsSaverLoader(name, params, self.context.session)
        self.filename = filename
        self.load_dir = load_dir
        self.save_dir = os.path.join(RL.logger.get_dir(), save_sub_dir)
        if self.load_dir is None:
            self.load_dir = self.context.load_model_dir

    def load_model(self):
        if self.load_dir is not None:
            load_path = os.path.join(self.load_dir, self.filename)
            self.loader_saver.load(load_path)

    def save_model(self):
        if self.save_dir is not None:
            save_path = os.path.join(self.save_dir, self.filename)
            save_path_for_episode = os.path.join(self.save_dir, self.filename + str(self.runner.episode_id))
            self.loader_saver.save(save_path)
            self.loader_saver.save(save_path_for_episode)
            RL.logger.log(self.filename, "saved", level=RL.logger.DEBUG)

    def start(self):
        super().start()
        self.load_model()

    def post_episode(self, env_id_nos):
        if self.runner.episode_id % self.context.save_every == 0:
            self.save_model()

    def close(self):
        self.save_model()
