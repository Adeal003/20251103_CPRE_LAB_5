#include <lwip/tcp.h>
#include <xil_printf.h>
#include <ff.h>
#include <xtime_l.h>

#define SERVER_PORT 80

namespace FileServer {

static FATFS fatfs = {};
static bool fatfs_mounted = false;

// Implement a very simple HTTP REST API via LWIP
// Only supports GET & PUT/POST/PATCH requests.
// Only checks the first character of the method, skips to the next space, records path to newline, skips to double newline (end of headers), then copies body into file contents or returns file contents as body of response.
// Return codes: 404 - Not Found, 500 - Bad SD Card
enum HttpState {
	METHOD, // Next char is a method character
	METHOD_SKIP, // Skip to the end of the method name
	PATH, // Path until char is a space
	HEADER, // Process headers
	BODY, // Body until connection close.
};

enum HttpMethod {
	INVALID,
	GET, // 'G'
	POST, // 'P'
	REFORMAT, // 'R' -- Nonstandard method name, but cURL doesn't care. Wipes and reformats SD card.
};

#define MAX_PATH_LEN 128
#define MAX_HEADER_LEN 64
#define CONNECTION_TIMEOUT_XTIME COUNTS_PER_SECOND

struct Connection {
	struct tcp_pcb *tpcb;
	XTime lastAccessTime;
	HttpState state;
	HttpMethod method;
	bool connected;
	u32 body_length_remaining;
	char header[MAX_HEADER_LEN];
	u32 header_length;
	char path[MAX_PATH_LEN];
	u32 path_length;
	bool file_open;
	FIL file;
};
static Connection conn = {};

bool fatfs_ensure_mounted() {
	if (fatfs_mounted) return true;
	if (f_mount(&fatfs, "/", 0) == FR_OK) {
		fatfs_mounted = true;
		return true;
	}
	return false;
}

// len == -1 -> use strlen on body instead to get the length
static char header[] = "HTTP/1.1 XXX\r\n\r\n";
err_t response(struct tcp_pcb *tpcb, int code, const char* body, int len) {
	if (len == -1) len = strlen(body);

	// Build HTTP header
	header[9] = '0' + ((code / 100) % 10);
	header[10] = '0' + ((code / 10) % 10);
	header[11] = '0' + ((code / 1) % 10);

	// xil_printf("%s\n", header);

	// Send HTTP header
	// if (tcp_sndbuf(tpcb) < sizeof(header)) {
	// 	// xil_printf("Error tcp send buffer ran out of space for the header??\n");
	// 	tcp_output(tpcb); // Flush, blocks until fully sent
	// }
	tcp_write(tpcb, header, sizeof(header) - 1, 0); // Don't send null terminator
	tcp_output(tpcb); // Flush, blocks until fully sent

	// Send HTTP body
	tcp_write(tpcb, body, len, 0);
	tcp_output(tpcb); // Flush, blocks until fully sent

	tcp_close(tpcb);
	if (tpcb == conn.tpcb && conn.connected) {
		conn.connected = false;
	}
	return ERR_OK;
}

err_t reformat(struct tcp_pcb *tpcb) {
	if (fatfs_mounted) {
		f_unmount("/");
		fatfs_mounted = false;
	}

	char work[FF_MAX_SS * 4];
	if (f_mkfs("", FM_FAT32, 0, work, sizeof(work)) == FR_OK) {
		if (f_mount(&fatfs, "/", 1) == FR_OK) {
			fatfs_mounted = true;
			return response(tpcb, 200, "SUCCESS\n", -1);
		}
		return response(tpcb, 500, "ERROR: Could not remount SD card after reformatting. This shouldn't be possible because reformatting succeeded?\n", -1);
	}
	return response(tpcb, 500, "ERROR: Could not reformat the SD card. Is it plugged in and writeable?\n", -1);
}

err_t get(struct tcp_pcb *tpcb, const char* path) {
	static FIL file;
	static char buffer[256];
	if (f_open(&file, path, FA_OPEN_EXISTING | FA_READ) != FR_OK) {
		return response(tpcb, 500, "ERROR: Could not open file for reading\n", -1);
	}

	const char header[] = "HTTP/1.1 200\r\n\r\n";
	tcp_write(tpcb, header, sizeof(header) - 1, 1); // Don't send null terminator

	while (1) {
		UINT bytes_read;
		if (f_read(&file, buffer, sizeof(buffer), &bytes_read) != FR_OK) {
			const char error[] = "ERROR: Failed reading file after opening";
			tcp_write(tpcb, error, sizeof(error) - 1, 0);
			break;
		}
		if (bytes_read == 0) break;
	
		if (tcp_sndbuf(tpcb) < bytes_read) {
			tcp_output(tpcb);
		}
		tcp_write(tpcb, buffer, bytes_read, 1);
		
		if (bytes_read < sizeof(buffer)) break;
	}
	f_close(&file);
	tcp_output(tpcb);

	tcp_close(tpcb);
	if (tpcb == conn.tpcb && conn.connected) {
		conn.connected = false;
	}
	return ERR_OK;
}

err_t list(struct tcp_pcb *tpcb, const char* path) {
	static DIR dir;
	static FILINFO info;

	if (f_opendir(&dir, path) != FR_OK) {
		return response(tpcb, 500, "ERROR: Could not open directory\n", -1);
	}

	const char header[] = "HTTP/1.1 200\r\n\r\n";
	tcp_write(tpcb, header, sizeof(header) - 1, 1); // Don't send null terminator

	while (1) {
		if (f_readdir(&dir, &info) != FR_OK) {
			const char error[] = "ERROR: Failed reading directory entry";
			tcp_write(tpcb, error, sizeof(error) - 1, 0);
			break;
		}

		if (info.fname[0] == 0) break;

		u32 len = strlen(info.fname);
		if (tcp_sndbuf(tpcb) < (len+1)) {
			tcp_output(tpcb);
		}
		tcp_write(tpcb, info.fname, len, 1);
		tcp_write(tpcb, "\n", 1, 1);
	}
	f_closedir(&dir);
	tcp_output(tpcb);

	tcp_close(tpcb);
	if (tpcb == conn.tpcb && conn.connected) {
		conn.connected = false;
	}
	return ERR_OK;
}

err_t recv_callback(void *arg, struct tcp_pcb *tpcb,
                               struct pbuf *p, err_t err)
{
	/* do not read the packet if we are not in ESTABLISHED state */
	if (!p) {
		if (tpcb == conn.tpcb && conn.connected) {
			conn.connected = false;
		}
		tcp_close(tpcb);
		tcp_recv(tpcb, NULL);
		return ERR_OK;
	}

	if (p->tot_len != p->len) {
		// xil_printf("Chained pbufs are not supported yet. Does this ever happen??\n");
		tcp_abort(tpcb);
		return ERR_ABRT;
	}

	/* indicate that the packet has been received */
	tcp_recved(tpcb, p->len);

	const char* workptr = (const char*) p->payload;

	// xil_printf("Processing TCP chunk: ");
	// for (int i = 0; i < p->len; i++) {
	// 	outbyte(workptr[i]);
	// }
	// outbyte('\n');
	
	for (int i = 0; i < p->len;i++) {
		// GET /file HTTP/1.1
		// ....
		// <blank line>
		// body

		// Skip \r, please except for the body section
		if ((conn.state != BODY) && (workptr[i] == '\r')) continue;

		switch (conn.state) {
			case METHOD: {
				// Grab first letter of method
				switch (workptr[i]) {
					case 'G': conn.method = GET; break;
					case 'P': conn.method = POST; break;
					case 'R': conn.method = REFORMAT; break;
					default:
						pbuf_free(p);
						return response(tpcb, 400, "ERROR: Invalid method, supported methods are GET, POST, and REFORMAT\n", -1);
				}
				// xil_printf("Determined method: %d\n", conn.method);
				conn.state = METHOD_SKIP;
				break;
			}
			case METHOD_SKIP: {
				// Skip to start of path
				if (workptr[i] == ' ') {
					conn.state = PATH;
				}
				break;
			}
			case PATH: {
				// Copy off path until next space or too long
				if (workptr[i] == ' ') {
					conn.path[conn.path_length] = 0;
					// xil_printf("Path found: %s\n", conn.path);
					conn.state = HEADER;
				} else if (conn.path_length == MAX_PATH_LEN) {
					pbuf_free(p);
					return response(tpcb, 400, "ERROR: Path too long\n", -1);
				} else {
					conn.path[conn.path_length++] = workptr[i];
				}


				break;
			}
			case HEADER: {
				// Copy off the header
				if (workptr[i] == '\n') {
					if (conn.header_length == 0) {
						conn.state = BODY;

						// xil_printf("Parsing body");

						if (conn.method == REFORMAT) {
							pbuf_free(p);
							return reformat(tpcb);
						}

						else if (conn.method == POST) {
							if (!fatfs_ensure_mounted()) {
								pbuf_free(p);
								return response(tpcb, 500, "ERROR: Could not mount SD card. Is it plugged in, writeable, and has a FAT32 partition? Try REFORMAT\n", -1);
							}

							if (conn.path_length == 0) {
								pbuf_free(p);
								return response(tpcb, 400, "ERROR: Invalid path for POST\n", -1);
							}

							if (conn.path[conn.path_length - 1] == '/') {
								// Trailing slash == mkdir
								// Remove trailing slash from path because f_mkdir doesn't like it
								conn.path[conn.path_length - 1] = 0;
								FRESULT res = f_mkdir(conn.path);
								if (res == FR_EXIST) {
									pbuf_free(p);
									return response(tpcb, 500, "ERROR: file exists\n", -1);
								} else if (res != FR_OK) {
									pbuf_free(p);
									return response(tpcb, 500, "ERROR: f_mkdir failed\n", -1);
								} else {
									pbuf_free(p);
									return response(tpcb, 200, "SUCCESS\n", -1);
								}
							}

							if (f_open(&conn.file, conn.path, FA_CREATE_ALWAYS | FA_WRITE) != FR_OK) {
								pbuf_free(p);
								return response(tpcb, 500, "ERROR: f_open failed\n", -1);
							}

							// fallthrough: file is created and opened, ready for f_write calls.
						}

						else if (conn.method == GET) {
							if (!fatfs_ensure_mounted()) {
								pbuf_free(p);
								return response(tpcb, 500, "ERROR: Could not mount SD card. Is it plugged in, writeable, and has a FAT32 partition? Try REFORMAT\n", -1);
							}

							if (conn.path_length == 0) {
								conn.path[0] = '/';
								conn.path[1] = 0;
								conn.path_length = 1;
							}

							// xil_printf("Reading %d: %s\n", strlen(conn.path), conn.path);

							FILINFO info;
							FRESULT res = f_stat(conn.path, &info);
							if (res == FR_NO_FILE) {
								pbuf_free(p);
								return response(tpcb, 404, "ERROR: File not found\n", -1);
							} else if (res != FR_OK && res != FR_INVALID_NAME) {
								pbuf_free(p);
								return response(tpcb, 500, "ERROR: f_stat failed\n", -1);
							} else if ((res == FR_INVALID_NAME) || (info.fattrib & AM_DIR)) {
								// `/` fails f_stat with FR_INVALID_NAME, so assume those are directories
								pbuf_free(p);
								return list(tpcb, conn.path);
							} else {
								pbuf_free(p);
								return get(tpcb, conn.path);
							}
						}
					} else {
						// Process header line here
						conn.header[conn.header_length] = 0;
						// xil_printf("Header found with length %d: %s\n", conn.header_length, conn.header);

						if (strncmp("Content-Length: ", conn.header, sizeof("Content-Length: ")-1) == 0) {
							// xil_printf("Found content length!");
							conn.body_length_remaining = atoi(&conn.header[16]);
						}

						conn.header_length = 0;
					}
				} else if (conn.header_length < MAX_HEADER_LEN) {
					conn.header[conn.header_length++] = workptr[i];
				}
				break;
			}
			case BODY: {
				// If we are here, we are in a POST request with a file opened.
				// xil_printf("POST: %d bytes remaining\n", conn.body_length_remaining);
	
				UINT bytes_written;
				if (f_write(&conn.file, &workptr[i], p->len - i, &bytes_written) != FR_OK) {
					pbuf_free(p);
					return response(tpcb, 500, "ERROR: f_write failed\n", -1);
				}
				// f_sync(&conn.file);
				// xil_printf("POST: wrote %d/%d bytes of %d/%d\n", bytes_written, conn.body_length_remaining, p->len, p->tot_len);

				i += bytes_written - 1;
				conn.body_length_remaining -= bytes_written;
				if (conn.body_length_remaining == 0) {
					f_close(&conn.file);
					pbuf_free(p);
					return response(tpcb, 200, "SUCCESS\n", -1);
				}
				break;
			}
		}
	}

	/* free the received pbuf */
	pbuf_free(p);

	return ERR_OK;
}

err_t accept_callback(void *arg, struct tcp_pcb *newpcb, err_t err)
{
	// xil_printf("Connection incoming!\n");
	// Set the receive callback for this connection
	if (conn.connected) {
		XTime now;
		XTime_GetTime(&now);
		if (now > (conn.lastAccessTime + CONNECTION_TIMEOUT_XTIME)) {
			// xil_printf("Timeout elapsed, closing old connection!\n");
			conn.connected = false;
			tcp_abort(conn.tpcb);
		} else {
			// xil_printf("Connection refused!\n");
			// Only one connection allowed at a time, close new connections while old one is open
			// const char error[] = "HTTP/1.1 500 Server Busy\n\nFile Transfer Server only supports a single connection";
			// tcp_write(newpcb, error, sizeof(error), 0);
			// tcp_output(newpcb);
			tcp_abort(newpcb);
		}

		return ERR_ABRT;
	}

	conn = {};
	conn.tpcb = newpcb;
	conn.connected = true;
	tcp_err(newpcb, [](void* arg, err_t err){
		conn.connected = false;
	});
	tcp_recv(newpcb, recv_callback);

	// xil_printf("Connection accepted!\n");
	return ERR_OK;
}

#define FCHECK_ERROR(fn)                           \
  if (FRESULT ret = fn; ret != FR_OK) {            \
    std::cout << "FAIL " #fn " " << ret << '\n';   \
    return 1;                                      \
  } else {                                         \
    std::cout << "OK " #fn "\n";                   \
  }

void app_init() {
	if (f_mount(&fatfs, "/", 1) == FR_OK) {
		fatfs_mounted = true;		
	}

	err_t err;

	// ref Vitis LWIP echo server example
	// Allocate a new TCP stack
	tcp_pcb* pcb = tcp_new_ip_type(IPADDR_TYPE_V4);
	if (!pcb) {
		// xil_printf("Error creating PCB. Out of Memory\n");
		return;
	}

	// Bind to all configured IPV4 addresses
	err = tcp_bind(pcb, IP4_ADDR_ANY, SERVER_PORT);
	if (err != ERR_OK) {
		// xil_printf("Unable to bind to port %d: err = %d\n\r", SERVER_PORT, err);
		return;
	}

	// Listen for connections
	pcb = tcp_listen(pcb);
	if (!pcb) {
		// xil_printf("Out of memory while tcp_listen\n\r");
		return;
	}

	// Accept connections using accept_callback
	tcp_accept(pcb, accept_callback);

	xil_printf("HTTP file server started!\n");
}

}