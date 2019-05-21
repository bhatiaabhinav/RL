import tensorflow as tf


class Context:
    """These should be like physical constants. Cannot change.
    """


class Genome:
    """All parameter which builds the brain. Design them in such a way that genome->phenome->fitness functions are smooth"""
    def __init__(self):
        pass


class Brain:
    def __init__(self, genome: Genome, features, labels):
        self.genome = genome
        self.accuracy = 0

    @property
    def fitness(self):
        """a function of time"""
        return self.accuracy


class GenomeOptimizer:
    """Typically evolution"""
    def __init__(self):
        pass
