# File Transfer Server

## Starting
Start the file server with `run

## List Files
Make a GET request to the folder path:
curl 192.168.1.2

## Read File
Make a GET request to the file path:
curl 192.168.1.2/README.md

## Write File
Make a POST request to the file path:
// From file
curl 192.168.1.2/README.md -X POST --data-binary "@README.md"
// From string
curl 192.168.1.2/Hello -X POST --data-binary "Hello World"

## Make directory
Make a POST request to the directory ending with `/`
curl 192.168.1.2/dir/ -X POST

## Reformat SD card
Make a REFORMAT request to any path
curl 192.168.1.2 -X REFORMAT
