# oneweektutorial2017
Scripts used for a tutorial on transfer learning for image classification for NERD's //oneweek

## What's in here?
- The slides from the 7/14/17 talk.
- The scripts used for the hotel identification use case.

If you're looking for the full walkthrough for the Hangman example from the talk, see [this git repo](https://github.com/Azure/Hangman) instead.

## What do I need to run the scripts?

I used the following; edits may be necessary if you use different versions
- An NC6-series (GPU) Deep Learning toolkit for the DSVM Azure resource
- A copy of a [pretrained AlexNet](https://mawahstorage.blob.core.windows.net/aerialimageclassification/models/AlexNet_cntk2beta15.model). (Note: this file is > 200 MB.)
- CNTK 2.0 installed in a Python 3.5 environment with the script-based CNTK installer
- The following other Python packages:
   - `argparse`
   - `selenium` (with `geckodriver` installed to browse image search engines using Firefox)
   - `pillow`
   - `argparse`
   - `pandas`

You can get instructions for running each script by typing `python <script-path> -h`.
