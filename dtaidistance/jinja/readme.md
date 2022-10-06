DTAIDistance source code templates
==================================

There is a lot of repetitive code in the DTAIDistance toolbox that
is not easily dealt with in code (e.g. because the linking with C, because
variations are required for different techniques). Therefore, we 
generate some methods from templates.

Instructions
------------

Run `make` in the `jinja` directory.

Dependencies
------------

We make use of the `jinja2` Python package to generate the source code files.

