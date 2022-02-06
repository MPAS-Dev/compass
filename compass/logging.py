import inspect
import functools


def log_method_call(method, logger):
    """
    Log the module path and file path of a call to a method, e.g.::

      compass calling: compass.landice.tests.dome.decomposition_test.DecompositionTest.run()
        in /turquoise/usr/projects/climate/mhoffman/mpas/compass/compass/landice/tests/dome/decomposition_test/__init__.py

    Parameters
    ----------
    method : method
        The method of a class that will be run immediately following this call

    logger: logging.Logger
        The logger to log the method path and file path to
    """
    if not inspect.ismethod(method):
        raise ValueError('The "method" argument must be a method')

    method_name = method.__name__
    child_class = method.__self__.__class__
    child_location = f'{child_class.__module__}.{child_class.__name__}'
    actual_class = None
    for cls in inspect.getmro(child_class):
        if method.__name__ in cls.__dict__:
            actual_class = cls
            break
    actual_location = f'{actual_class.__module__}.{actual_class.__name__}'
    try:
        class_file = inspect.getfile(actual_class)
    except TypeError:
        class_file = None
    logger.info(f'compass calling: {child_location}.{method_name}()')
    if child_location != actual_location:
        logger.info(f'  inherited from: {actual_location}.{method_name}()')
    if class_file is not None:
        logger.info(f'  in {class_file}')
