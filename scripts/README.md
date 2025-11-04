Do not run these scripts with the Vitis GUI open on the workspace you are interacting with. A workspace can only be accessed by one instance of Vitis at a time.

## Creating a Vitis Workspace
To generate a vitis workspace, run
```sh
./scripts/create_vitis -xsa_path path/to/vivado.xsa
```
A default simple XSA file which has a basic firmware-capable setup is available in `scripts/test_hardware.xsa`. This can be used for firmware-only testing.
If the exported Vivado XSA is modified, you should run this script again to generate a new workspace (symlinks are used to reference C source files, so nothing should be lost) and delete the old workspace when prompted.

This workspace can now be opened in Vitis 2020.1 via
```
/remote/Xilinx/2020.1/Vitis/2020.1/bin/vitis
```
Note, Vitis shows the welcome window even though an applcation project exists, close the welcome tab to see your application project.

## Flashing 
Flashing an FPGA can be done through the Vitis GUI manually via the build and launch buttons, or via a command-line script. The script is the preferred method unless Vitis is being used already.

To flash via the script, run
```sh
./scripts/flash_vitis
```

This will compile the project, flash it to the first connected zedboard, and wait for a few seconds while recording debug serial output.

## Testing the zedboard
If the zedboard hardware itself is suspect, just run
```sh
./scripts/test_vitis
```
alone, and you should see several commands being run, and the final UART output should end in `SUCCESS`. If it instead shows a line with `FAIL`, some portion was not working reading/writing to an SD card, check that a card exists and the write protect switch is off. If no UART output is recorded, then check the jumpers because the program was unable to run at all.
