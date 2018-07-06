#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<error.h>
#include<fcntl.h>
#include<poll.h>
#include<pthread.h>
#include<sys/types.h> 
#include<sys/stat.h> 
#include<sys/mman.h>
#include<unistd.h>

typedef unsigned char           uint8_t;
typedef unsigned short          uint16_t;
typedef unsigned int            uint32_t;
#define alt_u8  uint8_t
#define alt_u16 uint16_t
#define alt_u32 uint32_t
#define CSR_STATUS_REG                          (0x0)
#define CSR_CONTROL_REG                         (0x4)
#define CSR_DESCRIPTOR_FILL_LEVEL_REG           (0x8)
#define CSR_RESPONSE_FILL_LEVEL_REG             (0xC)
#define CSR_SEQUENCE_NUMBER_REG                 (0x10)
#define CSR_BUSY_MASK                           (1)
#define CSR_BUSY_OFFSET                         (0)
#define CSR_DESCRIPTOR_BUFFER_EMPTY_MASK        (1<<1)
#define CSR_DESCRIPTOR_BUFFER_EMPTY_OFFSET      (1)
#define CSR_DESCRIPTOR_BUFFER_FULL_MASK         (1<<2)
#define CSR_DESCRIPTOR_BUFFER_FULL_OFFSET       (2)
#define CSR_RESPONSE_BUFFER_EMPTY_MASK          (1<<3)
#define CSR_RESPONSE_BUFFER_EMPTY_OFFSET        (3)
#define CSR_RESPONSE_BUFFER_FULL_MASK           (1<<4)
#define CSR_RESPONSE_BUFFER_FULL_OFFSET         (4)
#define CSR_STOP_STATE_MASK                     (1<<5)
#define CSR_STOP_STATE_OFFSET                   (5)
#define CSR_RESET_STATE_MASK                    (1<<6)
#define CSR_RESET_STATE_OFFSET                  (6)
#define CSR_STOPPED_ON_ERROR_MASK               (1<<7)
#define CSR_STOPPED_ON_ERROR_OFFSET             (7)
#define CSR_STOPPED_ON_EARLY_TERMINATION_MASK   (1<<8)
#define CSR_STOPPED_ON_EARLY_TERMINATION_OFFSET (8)
#define CSR_IRQ_SET_MASK                        (1<<9)
#define CSR_IRQ_SET_OFFSET                      (9)
#define CSR_STOP_MASK                           (1)
#define CSR_STOP_OFFSET                         (0)
#define CSR_RESET_MASK                          (1<<1)
#define CSR_RESET_OFFSET                        (1)
#define CSR_STOP_ON_ERROR_MASK                  (1<<2)
#define CSR_STOP_ON_ERROR_OFFSET                (2)
#define CSR_STOP_ON_EARLY_TERMINATION_MASK      (1<<3)
#define CSR_STOP_ON_EARLY_TERMINATION_OFFSET    (3)
#define CSR_GLOBAL_INTERRUPT_MASK               (1<<4)
#define CSR_GLOBAL_INTERRUPT_OFFSET             (4)
#define CSR_STOP_DESCRIPTORS_MASK               (1<<5)
#define CSR_STOP_DESCRIPTORS_OFFSET             (5)
#define CSR_READ_FILL_LEVEL_MASK                (0xFFFF)
#define CSR_READ_FILL_LEVEL_OFFSET              (0)
#define CSR_WRITE_FILL_LEVEL_MASK               (0xFFFF0000)
#define CSR_WRITE_FILL_LEVEL_OFFSET             (16)
#define CSR_RESPONSE_FILL_LEVEL_MASK            (0xFFFF)
#define CSR_RESPONSE_FILL_LEVEL_OFFSET          (0)
#define CSR_READ_SEQUENCE_NUMBER_MASK           (0xFFFF)
#define CSR_READ_SEQUENCE_NUMBER_OFFSET         (0)
#define CSR_WRITE_SEQUENCE_NUMBER_MASK          (0xFFFF0000)
#define CSR_WRITE_SEQUENCE_NUMBER_OFFSET        (16)
#define IORD_8DIRECT(base,offset) *((volatile uint8_t*)((base)  + (offset)))
#define IORD_16DIRECT(base,offset) *((volatile uint16_t*)((base)  + (offset)))
#define IORD_32DIRECT(base,offset) *((volatile uint32_t*)((base)  + (offset)))
#define IOWR_8DIRECT(base,offset,value) *((volatile uint8_t*)((base)  + (offset))) = (value)
#define IOWR_16DIRECT(base,offset,value) *((volatile uint16_t*)((base)  + (offset))) = (value)
#define IOWR_32DIRECT(base,offset,value) *((volatile uint32_t*)((base)  + (offset))) = (value)
#define WR_CSR_STATUS(base, data)              IOWR_32DIRECT(base, CSR_STATUS_REG, data)
#define WR_CSR_CONTROL(base, data)             IOWR_32DIRECT(base, CSR_CONTROL_REG, data)
#define RD_CSR_STATUS(base)                    IORD_32DIRECT(base, CSR_STATUS_REG)
#define RD_CSR_CONTROL(base)                   IORD_32DIRECT(base, CSR_CONTROL_REG)
#define RD_CSR_DESCRIPTOR_FILL_LEVEL(base)     IORD_32DIRECT(base, CSR_DESCRIPTOR_FILL_LEVEL_REG)
#define RD_CSR_RESPONSE_FILL_LEVEL(base)       IORD_32DIRECT(base, CSR_RESPONSE_FILL_LEVEL_REG)
#define RD_CSR_SEQUENCE_NUMBER(base)           IORD_32DIRECT(base, CSR_SEQUENCE_NUMBER_REG)
#define DESCRIPTOR_READ_ADDRESS_REG                      (0x0)
#define DESCRIPTOR_WRITE_ADDRESS_REG                     (0x4)
#define DESCRIPTOR_LENGTH_REG                            (0x8)
#define DESCRIPTOR_CONTROL_STANDARD_REG                  (0xC)
#define DESCRIPTOR_SEQUENCE_NUMBER_REG                   (0xC)
#define DESCRIPTOR_READ_BURST_REG                        (0xE)
#define DESCRIPTOR_WRITE_BURST_REG                       (0xF)
#define DESCRIPTOR_READ_STRIDE_REG                       (0x10)
#define DESCRIPTOR_WRITE_STRIDE_REG                      (0x12)
#define DESCRIPTOR_READ_ADDRESS_HIGH_REG                 (0x14)
#define DESCRIPTOR_WRITE_ADDRESS_HIGH_REG                (0x18)
#define DESCRIPTOR_CONTROL_ENHANCED_REG                  (0x1C)
#define DESCRIPTOR_SEQUENCE_NUMBER_MASK                  (0xFFFF)
#define DESCRIPTOR_SEQUENCE_NUMBER_OFFSET                (0)
#define DESCRIPTOR_READ_BURST_COUNT_MASK                 (0x00FF0000)
#define DESCRIPTOR_READ_BURST_COUNT_OFFSET               (16)
#define DESCRIPTOR_WRITE_BURST_COUNT_MASK                (0xFF000000)
#define DESCRIPTOR_WRITE_BURST_COUNT_OFFSET              (24)
#define DESCRIPTOR_READ_STRIDE_MASK                      (0xFFFF)
#define DESCRIPTOR_READ_STRIDE_OFFSET                    (0)
#define DESCRIPTOR_WRITE_STRIDE_MASK                     (0xFFFF0000)
#define DESCRIPTOR_WRITE_STRIDE_OFFSET                   (16)
#define DESCRIPTOR_CONTROL_TRANSMIT_CHANNEL_MASK         (0xFF)
#define DESCRIPTOR_CONTROL_TRANSMIT_CHANNEL_OFFSET       (0)
#define DESCRIPTOR_CONTROL_GENERATE_SOP_MASK             (1<<8)
#define DESCRIPTOR_CONTROL_GENERATE_SOP_OFFSET           (8)
#define DESCRIPTOR_CONTROL_GENERATE_EOP_MASK             (1<<9)
#define DESCRIPTOR_CONTROL_GENERATE_EOP_OFFSET           (9)
#define DESCRIPTOR_CONTROL_PARK_READS_MASK               (1<<10)
#define DESCRIPTOR_CONTROL_PARK_READS_OFFSET             (10)
#define DESCRIPTOR_CONTROL_PARK_WRITES_MASK              (1<<11)
#define DESCRIPTOR_CONTROL_PARK_WRITES_OFFSET            (11)
#define DESCRIPTOR_CONTROL_END_ON_EOP_MASK               (1<<12)
#define DESCRIPTOR_CONTROL_END_ON_EOP_OFFSET             (12)
#define DESCRIPTOR_CONTROL_TRANSFER_COMPLETE_IRQ_MASK    (1<<14)
#define DESCRIPTOR_CONTROL_TRANSFER_COMPLETE_IRQ_OFFSET  (14)
#define DESCRIPTOR_CONTROL_EARLY_TERMINATION_IRQ_MASK    (1<<15)
#define DESCRIPTOR_CONTROL_EARLY_TERMINATION_IRQ_OFFSET  (15)
#define DESCRIPTOR_CONTROL_ERROR_IRQ_MASK                (0xFF<<16)
#define DESCRIPTOR_CONTROL_ERROR_IRQ_OFFSET              (16)
#define DESCRIPTOR_CONTROL_EARLY_DONE_ENABLE_MASK        (1<<24)
#define DESCRIPTOR_CONTROL_EARLY_DONE_ENABLE_OFFSET      (24)
#define DESCRIPTOR_CONTROL_GO_MASK                       (1<<31)
#define DESCRIPTOR_CONTROL_GO_OFFSET                     (31)
#define WR_DESCRIPTOR_READ_ADDRESS(base, data)           IOWR_32DIRECT(base, DESCRIPTOR_READ_ADDRESS_REG, data)
#define WR_DESCRIPTOR_WRITE_ADDRESS(base, data)          IOWR_32DIRECT(base, DESCRIPTOR_WRITE_ADDRESS_REG, data)
#define WR_DESCRIPTOR_LENGTH(base, data)                 IOWR_32DIRECT(base, DESCRIPTOR_LENGTH_REG, data)
#define WR_DESCRIPTOR_CONTROL_STANDARD(base, data)       IOWR_32DIRECT(base, DESCRIPTOR_CONTROL_STANDARD_REG, data)
#define WR_DESCRIPTOR_SEQUENCE_NUMBER(base, data)        IOWR_16DIRECT(base, DESCRIPTOR_SEQUENCE_NUMBER_REG, data)
#define WR_DESCRIPTOR_READ_BURST(base, data)             IOWR_8DIRECT(base, DESCRIPTOR_READ_BURST_REG, data)
#define WR_DESCRIPTOR_WRITE_BURST(base, data)            IOWR_8DIRECT(base, DESCRIPTOR_WRITE_BURST_REG, data)
#define WR_DESCRIPTOR_READ_STRIDE(base, data)            IOWR_16DIRECT(base, DESCRIPTOR_READ_STRIDE_REG, data)
#define WR_DESCRIPTOR_WRITE_STRIDE(base, data)           IOWR_16DIRECT(base, DESCRIPTOR_WRITE_STRIDE_REG, data)
#define WR_DESCRIPTOR_READ_ADDRESS_HIGH(base, data)      IOWR_32DIRECT(base, DESCRIPTOR_READ_ADDRESS_HIGH_REG, data)
#define WR_DESCRIPTOR_WRITE_ADDRESS_HIGH(base, data)     IOWR_32DIRECT(base, DESCRIPTOR_WRITE_ADDRESS_HIGH_REG, data)
#define WR_DESCRIPTOR_CONTROL_ENHANCED(base, data)       IOWR_32DIRECT(base, DESCRIPTOR_CONTROL_ENHANCED_REG, data)

#define REG_LEN      32UL
#define BUFFERSIZE   64UL
#define MAP_SIZE     4096UL
#define MAP_MASK     (MAP_SIZE - 1)
#define SETUP_IO_RES(e, f, s)    do { setup_io_res(&((e).phy_##f), &((e).mmaped_##f), s, "/dev/wr_fpga_ddr/"#f""); } while (0)
#define SETUP_IRQ_RES(e, mber)   do {\
                                    e.irq_fds[irq_##mber##_fd_index].fd = open("/dev/wr_fpga_ddr/"#mber"_irq", O_RDWR|O_SYNC);\
                                    if (e.irq_fds[irq_##mber##_fd_index].fd < 0)\
                                    {\
                                            printf("%s can't open!\r\n","/dev/wr_fpga_ddr/"#mber"_irq");\
                                    }\
                                    e.irq_fds[irq_##mber##_fd_index].events = POLLIN;\
                                 } while (0)

#define sgdma_extended_descriptor_packed __attribute__ ((packed, aligned(32)))
typedef struct {
  alt_u32 *read_address_low;
  alt_u32 *write_address_low;
  alt_u32 transfer_length;
  alt_u16 sequence_number;
  alt_u8  read_burst_count;
  alt_u8  write_burst_count;
  alt_u16 read_stride;
  alt_u16 write_stride;
  alt_u32 *read_address_high;
  alt_u32 *write_address_high;
  alt_u32 control;
} sgdma_extended_descriptor_packed sgdma_extended_descriptor;

typedef struct {
    alt_u32 phy_write_csr;
    alt_u32 phy_write_slave;
    alt_u32 phy_read_csr;
    alt_u32 phy_read_slave;
    alt_u32 phy_read_arm_buffer;
    alt_u32 phy_write_arm_buffer;
    alt_u32 mmaped_write_csr;
    alt_u32 mmaped_write_slave;
    alt_u32 mmaped_read_csr;
    alt_u32 mmaped_read_slave;
    alt_u32 mmaped_read_arm_buffer;
    alt_u32 mmaped_write_arm_buffer;
    alt_u32 dma_write_buffer_len;
    alt_u32 dma_read_buffer_len;
} wr_fpga_io_t;

enum {
    irq_write_fd_index = 0,
    irq_read_fd_index,
    IQR_MAX_INDEX,
};
typedef struct {
    unsigned long irq_count[IQR_MAX_INDEX];
    struct pollfd irq_fds[IQR_MAX_INDEX];
} wr_fpga_irq_t;

static alt_u32 get_alt_u32(char* path)
{
    alt_u32 num;
    int fd = 0;
    char buf[BUFFERSIZE] = {0};
    if ((fd = open(path, O_RDONLY|O_SYNC)) < 0)
    {
        printf("open error %s\r\n",path);
        return 0;
    }    
    while((read(fd, buf, BUFFERSIZE) > 0))  
    { 
        sscanf(buf, "%u", &num);
    }
    close(fd);
    
    return num;
}

static int setup_io_res(alt_u32* phy_addr, alt_u32* mapped_addr, alt_u32 len, char* path)
{
    int fd = 0;
    char buf[BUFFERSIZE] = {0};
    if ((fd = open(path, O_RDWR|O_SYNC)) < 0)
    {
        printf("open error :%s\r\n", path);
        return -1;
    }    
    while((read(fd, buf, BUFFERSIZE) > 0))  
    { 
        sscanf(buf, "%u", phy_addr);
    }  

    *mapped_addr = mmap(0, len, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (*mapped_addr == MAP_FAILED)
    {
        printf("mmap error\r\n");
        close(fd);
        return -1;
    }
    *mapped_addr = *mapped_addr + (*phy_addr & MAP_MASK);
    close(fd);
}

void enable_global_interrupt_mask (alt_u32 csr_base)
{
    alt_u32 temporary_control;
    temporary_control = RD_CSR_CONTROL(csr_base) | CSR_GLOBAL_INTERRUPT_MASK;
    WR_CSR_CONTROL(csr_base, temporary_control);
}

void disable_global_interrupt_mask (alt_u32 csr_base)
{
    alt_u32 temporary_control;
    temporary_control = RD_CSR_CONTROL(csr_base) & (CSR_GLOBAL_INTERRUPT_MASK ^ 0xFFFFFFFF);
    WR_CSR_CONTROL(csr_base, temporary_control);
}

int write_extended_descriptor (alt_u32 csr_base, alt_u32 descriptor_base, sgdma_extended_descriptor *descriptor)
{
    if ((RD_CSR_STATUS(csr_base) & CSR_DESCRIPTOR_BUFFER_FULL_MASK) != 0)
    {
        return -1;
    }
    WR_DESCRIPTOR_READ_ADDRESS(descriptor_base, (alt_u32)descriptor->read_address_low);
    WR_DESCRIPTOR_WRITE_ADDRESS(descriptor_base, (alt_u32)descriptor->write_address_low);
    WR_DESCRIPTOR_LENGTH(descriptor_base, descriptor->transfer_length);
    WR_DESCRIPTOR_SEQUENCE_NUMBER(descriptor_base, descriptor->sequence_number);
    WR_DESCRIPTOR_READ_BURST(descriptor_base, descriptor->read_burst_count);
    WR_DESCRIPTOR_WRITE_BURST(descriptor_base, descriptor->write_burst_count);
    WR_DESCRIPTOR_READ_STRIDE(descriptor_base, descriptor->read_stride);
    WR_DESCRIPTOR_WRITE_STRIDE(descriptor_base, descriptor->write_stride);
    WR_DESCRIPTOR_READ_ADDRESS_HIGH(descriptor_base, (alt_u32)descriptor->read_address_high);
    WR_DESCRIPTOR_WRITE_ADDRESS_HIGH(descriptor_base, (alt_u32)descriptor->write_address_high);
    WR_DESCRIPTOR_CONTROL_ENHANCED(descriptor_base, descriptor->control);
    return 0;
}

void write_fpga(wr_fpga_io_t card, alt_u32 dts)
{
    static sgdma_extended_descriptor write_descriptor;
    memset(&write_descriptor, 0, sizeof(write_descriptor));
    write_descriptor.transfer_length  = card.dma_write_buffer_len;
    write_descriptor.read_burst_count = card.dma_write_buffer_len;
    write_descriptor.control = DESCRIPTOR_CONTROL_GO_MASK|DESCRIPTOR_CONTROL_TRANSFER_COMPLETE_IRQ_MASK|DESCRIPTOR_CONTROL_EARLY_TERMINATION_IRQ_MASK;
    write_descriptor.read_address_low  = card.phy_write_arm_buffer;
    write_descriptor.write_address_low = dts;
    write_extended_descriptor(card.mmaped_write_csr, card.mmaped_write_slave, &write_descriptor); 
}
void read_fpga(wr_fpga_io_t card, alt_u32 src)
{
    static sgdma_extended_descriptor read_descriptor;
    memset(&read_descriptor, 0, sizeof(read_descriptor));
    read_descriptor.transfer_length  = card.dma_read_buffer_len;
    read_descriptor.read_burst_count = card.dma_read_buffer_len;
    read_descriptor.control = DESCRIPTOR_CONTROL_GO_MASK|DESCRIPTOR_CONTROL_TRANSFER_COMPLETE_IRQ_MASK|DESCRIPTOR_CONTROL_EARLY_TERMINATION_IRQ_MASK;
    read_descriptor.read_address_low  = src;
    read_descriptor.write_address_low = card.phy_read_arm_buffer;
    write_extended_descriptor(card.mmaped_read_csr, card.mmaped_read_slave, &read_descriptor); 
}

static void print_hex_str(const void* buf , unsigned int size)
{
    unsigned char* str = (unsigned char*)buf;
    char line[512] = {0};
    const size_t lineLength = 16; // 8或者32
    char text[24] = {0};
    char* pc;
    int textLength = lineLength;
    unsigned int ix = 0 ;
    unsigned int jx = 0 ;

    for (ix = 0 ; ix < size ; ix += lineLength) {
        sprintf(line, "%.8xh: ", ix);
// 打印16进制
        for (jx = 0 ; jx != lineLength ; jx++) {
            if (ix + jx >= size) {
                sprintf(line + (11 + jx * 3), "   "); // 处理最后一行空白
                if (ix + jx == size)
                    textLength = jx;  // 处理最后一行文本截断
            } else
                sprintf(line + (11 + jx * 3), "%.2X ", * (str + ix + jx));
        }
// 打印字符串
        {
            memcpy(text, str + ix, lineLength);
            pc = text;
            while (pc != text + lineLength) {
                if ((unsigned char)*pc < 0x20) // 空格之前为控制码
                    *pc = '.';                 // 控制码转成'.'显示
                pc++;
            }
            text[textLength] = '\0';
            sprintf(line + (11 + lineLength * 3), "; %s", text);
        }

        printf("%s\n", line);
    }
}
