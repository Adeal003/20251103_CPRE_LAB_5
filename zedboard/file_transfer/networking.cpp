#include <iostream>
#include <ff.h>
#include <lwip/tcp.h>
#include <lwip/init.h>
#include <lwip/sys.h>
#include <netif/xadapter.h>
#include <xemacps.h>
#include <xparameters.h>
#include <xtime_l.h>
#include <lwip/priv/tcp_priv.h>
#include <xscugic.h>

namespace FileServer {

void app_init();

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

static struct netif server_netif;
static ip_addr_t server_ip;
static ip_addr_t server_netmask;
static ip_addr_t server_gateway;
// Hardcoded MAC address (from LWIP Echo Server example), hopefully this is unique. Don't connect zedboards together with this.
static u8 server_macaddress[6] = { 0x00, 0x0a, 0x35, 0x00, 0x01, 0x02 };

void start_file_transfer_server() {
  // Initialize static connection configuration
  // Transform IP address to u32 and save in ip_addr_t*
  // IP_ADDR4(&server_ip, 192, 168, 2, 1);
  // IP_ADDR4(&server_netmask, 255, 255, 255, 0);
  // IP_ADDR4(&server_gateway, 192, 168, 2, 2);
  IP_ADDR4(&server_ip, 192, 168, 1, 2);
  IP_ADDR4(&server_netmask, 255, 255, 255, 0);
  IP_ADDR4(&server_gateway, 192, 168, 1, 1);

  xil_printf("IP:      %d.%d.%d.%d\n", ip4_addr1(&server_ip), ip4_addr2(&server_ip), ip4_addr3(&server_ip), ip4_addr4(&server_ip));
  xil_printf("Netmask: %d.%d.%d.%d\n", ip4_addr1(&server_netmask), ip4_addr2(&server_netmask), ip4_addr3(&server_netmask), ip4_addr4(&server_netmask));
  xil_printf("Gateway: %d.%d.%d.%d\n", ip4_addr1(&server_gateway), ip4_addr2(&server_gateway), ip4_addr3(&server_gateway), ip4_addr4(&server_gateway));
  xil_printf("MAC:     %02X:%02X:%02X:%02X:%02X:%02X\n", (u32)server_macaddress[0], (u32)server_macaddress[1], (u32)server_macaddress[2], (u32)server_macaddress[3], (u32)server_macaddress[4], (u32)server_macaddress[5]);

  // Initialize the global interrupt controller (required for LWIP)
  Xil_ExceptionInit();
	XScuGic_DeviceInitialize(XPAR_SCUGIC_0_DEVICE_ID);

	// Connect the interrupt controller interrupt handler to the hardware interrupt handling logic in the processor.
	Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_IRQ_INT,
			(Xil_ExceptionHandler)XScuGic_DeviceInterruptHandler,
			(void *)XPAR_SCUGIC_0_DEVICE_ID);

  lwip_init();

  // Initialize XEmac network interface
  xemac_add(&server_netif, &server_ip, &server_netmask, &server_gateway, server_macaddress, XPAR_XEMACPS_0_BASEADDR);
  netif_set_default(&server_netif);
  netif_set_up(&server_netif);

  app_init();

  // Enable interrupts globally (required for LWIP)
  Xil_ExceptionEnableMask(XIL_EXCEPTION_IRQ);

  XTime tNextTcpTmr = 0;
  XTime tNow = 0;
  while (true) {
    // LWIP requries timer callback to be called ~250ms apart, rough timing is ok.
    XTime_GetTime(&tNow);
    if (tNow > tNextTcpTmr) {
      tNextTcpTmr = tNow + (COUNTS_PER_SECOND/4); // ~250ms in the future
      // xil_printf("TCP TMR Callback @ %u %u\n", (u32)tNow, (u32)(tNow << 32));
      tcp_tmr();
    }

    xemacif_input(&server_netif);
  }
}

}