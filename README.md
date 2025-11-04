# CprE 487 / 587 ML C++ Framework
Written by: Matthew Dwyer

This repository contains the source code for the C++ ML framework.

## Navigation
TODO

## Building
Get started by running `make help`. This will display a list of make targets:
```shell
487/587 ML Framework Help:
        Usage: make <target> [params]

 Targets:
        build:          Builds the framework (same as 'make')
                        NOTE: header files (.h/.hpp) changes are not detected, please 'clean' first
        rebuild:        Performs a 'clean' then 'build'
        build_debug:    Same as 'build', but with without optimizations and debug information
        redebug:        Performs a 'clean' then 'debug' build
        clean:          Cleans all build artifacts
        update:         Checks for a framework update. If one is found, it is pulled
        submit:         Zips directory for submission
        help:           Shows this help message
```
To build the framework, run `make build`. To run the build binary, run `./build/ml`. This will run some basic checks to ensure that your framework is built correctly.

## Building for zedboard
From the framework folder, run `./scripts/create_vitis -xsa_path path/to/hardware.xsa`. It will create a Vitis workspace in `workspace` and compile the project. To just compile the project without regenerating the entire workspace, run `./scripts/flash_vitis`.

If you update the hardware xsa, just rerun the create script again and delete the workspace.

1) Flash the file transfer application to the zedboard and copy the data folder over
```sh
./scripts/file_transfer_vitis
```
Then, while the server script _is still running_, once it says that the HTTP server has started, run
```sh
./scripts/upload_data
```
in a separate terminal to reformat the SD card and upload the data folder. 

Then, kill the file transfer server on the zedboard script by pressing Ctrl-C in that terminal. Only one *_vitis script should be running at a time.

2) Create Vitis workspace for your project. There is a default xsa with no special hardware in `scripts/test_hardware.xsa`
```sh
./scripts/create_vitis -xsa_path path/to/hardware.xsa
```
This must be done only when the XSA changes

3) Build and flash the FPGA
```sh
./scripts/flash_vitis
```
Press Ctrl-C to stop uart output.
NOTE: Program on zedboard will hang on std::cout or xil_printf after killing this command and will stop running.

4) Once inference completes, the file transfer server (same as step #1 above) will start. See zedboard/file_transfer/README.md for the cURL api for accessing files. Again, if you kill `./scripts/flash_vitis`, the file transfer server will stop responding.

5) If code changes, run #3 again, if xsa changes, run #2 and #3 again.

## Help! Zedboard library header files are not found!
Make sure the `workspace` has been created using `./scripts/create_vitis ...` and make sure the C++ extension by Microsoft is installed and the Configuration Provider (bottom right corner) says `Zedboard`, not `Local`.