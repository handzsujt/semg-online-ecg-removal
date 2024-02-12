# semg-online-ecg-removal

This package contains a python algorithm for online filtering an sEMG-signal using a stationary wavelet transform.
The class `sEMGOnlineFilter` in `sEMG_online_filter.py` can be used as an interface for using the online sEMG ECG removal algorithm in your own project.
An example how to do this is given in the file `sEMG_simulation_multichannel.py`. There you can see how to filter your data using the `sEMGOnlineFilter` class.
It is also shown, how to plot the raw and filtered signals fast. Therefore, a recorded respiratory sEMG with airway pressure measurement is used.

In order to start the simulation, you have to run the `main` method in `main.py`.

## Requirements

The requirements can be installed with
``
pip install -r requirements.txt
``
.
