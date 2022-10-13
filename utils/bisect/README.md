Using `git bisect`
====================

The command `git bisect` is a handy tool for finding the first commit that
breaks a code in some way.  `git bisect run` can call a script that
succeeds when a given commit is "good" but fails when it is "bad".  The script
`bisect_step.py` provided here is one such script.

To further encapsulate the process of using `git bisect`, we provide a driver
script `bisect.py` that makes use of config options in a file similar to
`example.cfg`.

Instructions
------------

1. Copy `example.cfg` to the base of the branch:
   ```shell
   cp utils/bisect/example.cfg bisect.cfg
   ```
2. Modify the config options with the appropriate "good" and "bad" E3SM commit hash
or tag.

3. Modify the various paths and commands as needed.

4. In a batch job or interactive session on a compute node, run:
   ```shell
   ./utils/bisect/bisect.py -f bisect.cfg
   ```

Note
----

Since the code will be compiled on a compute node, any baseline use for
comparison should also be built on a compute node.  Otherwise, you may get
non-bit-for-bit results simply because of where the code was compiled.  This
has been seen with Intel on Anvil.