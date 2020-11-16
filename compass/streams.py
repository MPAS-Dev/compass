from lxml import etree
from copy import deepcopy
from importlib import resources


def read(package, streams_filename, tree=None):
    """
    Parse the given streams file

    Parameters
    ----------
    package : Package
        The package name or module object that contains ``namelist``

    streams_filename : str
        The name of the streams file to read from

    tree : lxml.etree, optional
        An existing set of streams to add to or modify

    Returns
    -------
    tree : lxml.etree
        A tree of XML data describing MPAS i/o streams with the content from the
        given streams file
    """
    new_tree = etree.fromstring(resources.read_text(package, streams_filename))

    if tree is None:
        tree = new_tree
    else:
        streams = next(tree.iter('streams'))
        new_streams = next(new_tree.iter('streams'))

        for new_stream in new_streams:
            _update_element(new_stream, streams)

    return tree


def generate(config, tree, step_work_dir, core, mode='forward'):
    """
    Writes out a steams file in the ``work_case_dir``

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options used determine the name of the streams file and
        the default streams and options

    tree : lxml.etree
        A tree of XML data describing MPAS i/o streams with the content from
        one or more streams files parsed with :py:func:`compass.streams.read()`

    step_work_dir : str
        The path for the work directory for the step that this namelist is being
        generated for

    core : str
        The name of the MPAS core ('ocean', 'landice', etc.)

    mode : {'init', 'forward'}, optional
        The mode that the model will run in
    """
    out_name = 'streams.{}'.format(core)

    defaults_filename = config.get('streams', mode)
    out_filename = '{}/{}'.format(step_work_dir, out_name)

    defaults_tree = etree.parse(defaults_filename)

    defaults = next(defaults_tree.iter('streams'))
    streams = next(tree.iter('streams'))

    for stream in streams:
        _update_defaults(stream, defaults)

    # remove any streams that aren't requested
    for default in defaults:
        found = False
        for stream in streams:
            if stream.attrib['name'] == default.attrib['name']:
                found = True
                break
        if not found:
            defaults.remove(default)

    _write(defaults_tree, out_filename)


def _write(streams, out_filename):
    """ write the streams XML data to the file """

    with open(out_filename, 'w') as stream_file:

        stream_file.write('<streams>\n')

        # Write out all immutable streams first
        for stream in streams.findall('immutable_stream'):
            stream_name = stream.attrib['name']

            stream_file.write('\n')
            stream_file.write('<immutable_stream name="{}"'.format(stream_name))
            # Process all attributes on the stream
            for attr, val in stream.attrib.items():
                if attr.strip() != 'name':
                    stream_file.write('\n                  {}="{}"'.format(
                        attr, val))

            stream_file.write('/>\n')

        # Write out all immutable streams
        for stream in streams.findall('stream'):
            stream_name = stream.attrib['name']

            stream_file.write('\n')
            stream_file.write('<stream name="{}"'.format(stream_name))

            # Process all attributes
            for attr, val in stream.attrib.items():
                if attr.strip() != 'name':
                    stream_file.write('\n        {}="{}"'.format(attr, val))

            stream_file.write('>\n\n')

            # Write out all contents of the stream
            for tag in ['stream', 'var_struct', 'var_array', 'var']:
                for child in stream.findall(tag):
                    child_name = child.attrib['name']
                    if tag == 'stream' and child_name == stream_name:
                        # don't include the stream itself
                        continue
                    if 'packages' in child.attrib.keys():
                        package_name = child.attrib['packages']
                        entry = '    <{} name="{}" packages="{}"/>\n' \
                                ''.format(tag, child_name, package_name)
                    else:
                        entry = '    <{} name="{}"/>\n'.format(tag, child_name)
                    stream_file.write(entry)

            stream_file.write('</stream>\n')

        stream_file.write('\n')
        stream_file.write('</streams>\n')


def _update_element(new_child, elements):
    """
    add the new child/grandchildren or add/update attributes if they exist
    """
    if 'name' not in new_child.attrib:
        return

    name = new_child.attrib['name']
    found = False
    for child in elements:
        if child.attrib['name'] == name:
            found = True
            if child.tag != new_child.tag:
                raise ValueError('Trying to update streams data with '
                                 'inconsistent tags.')

            # copy the attributes
            for attr, value in new_child.attrib.items():
                child.attrib[attr] = value

            # copy or add the grandchildren's contents
            for new_grandchild in new_child:
                _update_element(new_grandchild, child)

    if not found:
        # add a deep copy of the element
        elements.append(deepcopy(new_child))


def _update_defaults(new_child, defaults):
    """
    Update a stream or its children (sub-stream, var, etc.) starting from the
    defaults or add it if it's new.
    """
    if 'name' not in new_child.attrib:
        return

    name = new_child.attrib['name']
    found = False
    for child in defaults:
        if child.attrib['name'] == name:
            found = True
            if child.tag != new_child.tag:
                raise ValueError('Trying to update streams data with '
                                 'inconsistent tags.')

            # copy the attributes
            for attr, value in new_child.attrib.items():
                child.attrib[attr] = value

            if len(new_child) > 0:
                # we don't want default grandchildren
                for grandchild in child:
                    child.remove(grandchild)

            # copy or add the grandchildren's contents
            for new_grandchild in new_child:
                _update_defaults(new_grandchild, child)

    if not found:
        # add a deep copy of the element
        defaults.append(deepcopy(new_child))
