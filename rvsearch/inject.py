#Injection and recovery class.

class Inject(object):
    """Class to perform injection and recovery on a planetary system. Takes an instantiation of
    the Search class, once the planet detection loop has been completed.

    Args:
        post: Posterior, often taken from a completed instatiation of the Search class,
        once planet search has been completed.
    """
    def __init__(self, post):
        self.basepost = post
