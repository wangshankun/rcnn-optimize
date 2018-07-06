#include"user_so_wr_fpga_ddr.h"

wr_fpga_io_t  wr_fpga_io;
wr_fpga_irq_t wr_fpga_irq;
sgdma_extended_descriptor write_descriptor;
sgdma_extended_descriptor read_descriptor;

int write_ret;
struct pollfd write_irq_fds[1];
int read_ret;
struct pollfd read_irq_fds[1];

int hardrock_read_fpga(char* ptr, unsigned int fpga_offset, unsigned int len)
{
    read_fpga(wr_fpga_io, fpga_offset);
    
    read_ret = poll(read_irq_fds, 1, -1);
    if(read_ret < 0)
    {  
        printf("poll error!\n");
        return -1;
    }
    if(read_irq_fds[0].revents & POLLIN)
    {
        if(read(read_irq_fds[0].fd, &wr_fpga_irq.irq_count[irq_read_fd_index], sizeof(unsigned long)) > 0)
        {
            memcpy(ptr, wr_fpga_io.mmaped_read_arm_buffer, len);
            return 0;
        }
    }
}

int hardrock_write_fpga(char* ptr, unsigned int fpga_offset, unsigned int len)
{
    memcpy(wr_fpga_io.mmaped_write_arm_buffer, ptr, len);
    write_fpga(wr_fpga_io, fpga_offset);
    
    write_ret = poll(write_irq_fds, 1, -1);
    if(write_ret < 0)
    {  
        printf("poll error!\n");
        return -1;
    }
    if(write_irq_fds[0].revents & POLLIN)
    {
        if(read(write_irq_fds[0].fd, &wr_fpga_irq.irq_count[irq_write_fd_index], sizeof(unsigned long)) > 0)
        {
            return 0;
        }
    }
}
void hardrock_init(void)
{
    wr_fpga_io.dma_write_buffer_len = get_alt_u32("/sys/module/wr_fpga_ddr/parameters/dma_len");
    wr_fpga_io.dma_read_buffer_len = get_alt_u32("/sys/module/wr_fpga_ddr/parameters/dma_len");
    
    SETUP_IO_RES(wr_fpga_io, write_csr, REG_LEN);
    SETUP_IO_RES(wr_fpga_io, write_slave, REG_LEN);
    SETUP_IO_RES(wr_fpga_io, read_csr, REG_LEN);
    SETUP_IO_RES(wr_fpga_io, read_slave, REG_LEN);
    SETUP_IO_RES(wr_fpga_io, read_arm_buffer, wr_fpga_io.dma_read_buffer_len);
    SETUP_IO_RES(wr_fpga_io, write_arm_buffer, wr_fpga_io.dma_write_buffer_len);

    SETUP_IRQ_RES(wr_fpga_irq, write);
    SETUP_IRQ_RES(wr_fpga_irq, read);

    enable_global_interrupt_mask(wr_fpga_io.mmaped_write_csr);
    enable_global_interrupt_mask(wr_fpga_io.mmaped_read_csr);

    write_irq_fds[0].fd     = wr_fpga_irq.irq_fds[irq_write_fd_index].fd;
    write_irq_fds[0].events = POLLIN;
    read_irq_fds[0].fd      = wr_fpga_irq.irq_fds[irq_read_fd_index].fd;
    read_irq_fds[0].events  = POLLIN;
}

void hardrock_exit(void)
{
    munmap(wr_fpga_io.mmaped_write_csr  , REG_LEN);
    munmap(wr_fpga_io.mmaped_write_slave, REG_LEN);
    munmap(wr_fpga_io.mmaped_read_csr   , REG_LEN);
    munmap(wr_fpga_io.mmaped_read_slave , REG_LEN);
    munmap(wr_fpga_io.mmaped_read_arm_buffer , wr_fpga_io.dma_read_buffer_len);
    munmap(wr_fpga_io.mmaped_write_arm_buffer, wr_fpga_io.dma_write_buffer_len);
    
    int i = 0; 
    for(i = irq_write_fd_index; i < IQR_MAX_INDEX; i++)  
    { 
        close(wr_fpga_irq.irq_fds[i].fd);
   
    }
}