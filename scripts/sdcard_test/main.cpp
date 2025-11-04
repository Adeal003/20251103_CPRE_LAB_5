#include <iostream>
#include <ff.h>

// This script runs a quick test to verify the SD card is functional.
// It will reformat any inserted sd card back to a single FAT32 partition
// and write a single file `/ test` with the contents `K` and read it back.

#define FCHECK_ERROR(fn)                           \
  if (FRESULT ret = fn; ret != FR_OK) {            \
    std::cout << "FAIL " #fn " " << ret << '\n'; \
    return 1;                                      \
  } else {                                         \
    std::cout << "OK " #fn "\n";                      \
  }

static FATFS fatfs = {};

extern "C"
int main() {
	std::cout << "Running SD Card Test!\n";

  char work[FF_MAX_SS * 4];
  FCHECK_ERROR(f_mkfs("", FM_FAT32, 0, work, sizeof(work)));
    
  FCHECK_ERROR(f_mount(&fatfs, "/", 1));
  
  {
    FIL file;
    FCHECK_ERROR(f_open(&file, "/test", FA_READ | FA_WRITE | FA_CREATE_ALWAYS));

    char c = 'K';
    UINT bytes_written = 0;
    FCHECK_ERROR(f_write(&file, &c, 1, &bytes_written));

    if (bytes_written != 1) {
      std::cout << "Failed write length " << bytes_written << '\n';
      return 1;
    }

    FCHECK_ERROR(f_close(&file));
  }

  {
    FIL file;
    FCHECK_ERROR(f_open(&file, "/test", FA_READ));
      
    char c = 0;
    UINT bytes_read = 0;
    FCHECK_ERROR(f_read(&file, &c, 1, &bytes_read));
      
    if (bytes_read != 1) {
      std::cout << "Failed read length " << bytes_read << '\n';
      return 1;
    }

    if (c != 'K') {
      std::cout << "Failed read value " << (int)c << '\n';
      return 1;
    }

    FCHECK_ERROR(f_close(&file));   
  }

	std::cout << "SUCCESS\n";
  return 0;
}
