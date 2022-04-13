.. versastat documentation master file, created by
   sphinx-quickstart on Mon Jan  8 11:29:00 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

versastat-utils
===============

A python library for VersaSTAT automation.

Instruments can be controlled via the VersaSTATControl library.
From the SDK docs, it seems that a series of experiments can be set up (argument lists are passed as comma-separated strings...) and run asynchronously.
It should be possible to live-stream the data from the instrument via calls to Experiment.GetData* to build a real-time (interactive?) UI.

Motion controller: XCD controller by nanomotion.
This appears to be https://www.nanomotion.com/wp-content/uploads/2014/05/XCD-controller-user-manual.pdf

.. toctree::
   :maxdepth: 2
	      
   api

..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
