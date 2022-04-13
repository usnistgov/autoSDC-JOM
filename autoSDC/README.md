a-sdc
=====

automation tools for autonomous scanning droplet cell electrochemical characterization


Autonomous scanning droplet cell for data-driven experimental alloy design

We are developing an autonomous scanning droplet cell (ASDC) capable of on-demand electrodeposition and real-time electrochemical characterization for investigating multicomponent alloy systems for favorable corrosion-resistance properties.
The ASDC consists of a millimeter-scale electrochemical cell and an array of programmable pumps that can be used to electrodeposit an alloy film and immediately acquire polarization curves to obtain electrochemical quantities of interest, such as the passive current density and oxide breakdown potentials.
We model these quantities using Gaussian Process regression to select the most informative series of alloys to synthesize and characterize, continuously updating the model as new electrochemical data is acquired.
Our initial studies focus on systems that are likely to form corrosion-resistant metallic glasses (MGs) and single-phase multi-principle element alloys (MPEAs).
Looking forward, we see opportunities to incorporate prior predictions for MG and MPEA stability and real-time feedback in the form of on-demand CALPHAD computations.

asdc.control
------------
Instruments can be controlled via the VersaSTATControl library.
From the SDK docs, it seems that a series of experiments can be set up (argument lists are passed as comma-separated strings...) and run asynchronously.
It should be possible to live-stream the data from the instrument via calls to Experiment.GetData* to build a real-time (interactive) UI by pushing data through a websocket to a holoviews server.


asdc.position
-------------
Motion controller: XCD controller by nanomotion.
Appears to be https://www.nanomotion.com/wp-content/uploads/2014/05/XCD-controller-user-manual.pdf
