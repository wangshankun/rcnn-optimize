#include<linux/fs.h>
#include<linux/mm.h>
#include<linux/poll.h>
#include<linux/types.h>
#include<linux/errno.h>
#include<linux/string.h> 
#include<linux/miscdevice.h>

static unsigned int dma_len = 2*1024*1024;
module_param(dma_len, uint, S_IRUGO);
#define DMA_LEN  dma_len

#define WR_CRIT     1
#define WR_WARNING  2
#define WR_INFO     3

#define PRINTK_DEBUG(num, fmt, args...) do{if(num <= debug_on)printk("\033[46;31m[%s:%d]\033[0m "#fmt"\r\n", __func__, __LINE__, ##args);}while(0)

#define CSR_IRQ_SET_MASK                       (1<<9)
#define CSR_IRQ_SET_OFFSET                     (9)
#define CSR_STATUS_REG                         (0x0)
#define IOWR_32DIRECT(base,offset,value)       *((volatile uint32_t*)((base)  + (offset))) = (value)
#define WR_CSR_STATUS(base, data)              IOWR_32DIRECT(base, CSR_STATUS_REG, data)

#define MEM_OPS_API_DEFINE(mber)  \
    static int sp_uio_##mber##_read(struct file *file, char __user *buf, size_t size, loff_t *ppos)\
    {\
        return read_device_mem(wr_fpga_io.phy_##mber, buf, size, ppos);\
    }\
    static int sp_uio_##mber##_mmap(struct file *file, struct vm_area_struct *vma)\
    {\
        return mmap_device_mem(wr_fpga_io.phy_##mber, vma);\
    }\
    struct file_operations phy_##mber##_fops = \
    {\
        .owner    =   THIS_MODULE,\
        .open     =   sp_uio_open,\
        .read     =   sp_uio_##mber##_read,\
        .release  =   sp_uio_release,\
        .mmap     =   sp_uio_##mber##_mmap,\
    };\
    struct miscdevice mber##_dev = \
    {\
        .minor    =   MISC_DYNAMIC_MINOR,\
        .fops     =   &phy_##mber##_fops,\
        .name     =   ""#mber"",\
        .nodename =   "wr_fpga_ddr/"#mber""\
    };\

#define IRQ_OPS_API_DEFINE(mber)  \
    static unsigned long mber##_irq_count = 0;\
    static DECLARE_WAIT_QUEUE_HEAD(wait_##mber##_irq);\
    static volatile int mber##_irq_ev = 0;\
    static unsigned sp_uio_of_##mber##_irq_poll(struct file *file, poll_table *wait)\
    {\
        unsigned int mask = 0;\
        poll_wait(file, &wait_##mber##_irq, wait);\
        if (mber##_irq_ev)\
        {\
            mask |= POLLIN | POLLRDNORM;\
        }\
        return mask;\
    }\
    ssize_t sp_uio_of_##mber##_irq_read(struct file *file, char __user *buf, size_t size, loff_t *ppos)\
    {\
        wait_event_interruptible(wait_##mber##_irq, mber##_irq_ev);\
        put_user(mber##_irq_count, (unsigned long __user *)buf);\
        mber##_irq_ev = 0;\
        return sizeof(unsigned long);\
    }\
    static irqreturn_t mber##_irq_fuc(int irq, void *dev_id)\
    {\
        clear_irq(wr_fpga_io.kvir_##mber##_csr);\
        mber##_irq_ev = 1;\
        mber##_irq_count++;\
        wake_up_interruptible(&wait_##mber##_irq);\
        return IRQ_HANDLED;\
    }\
    struct file_operations of_##mber##_irq_fops = \
    {\
        .owner    =   THIS_MODULE,\
        .open     =   sp_uio_open,\
        .read     =   sp_uio_of_##mber##_irq_read,\
        .release  =   sp_uio_release,\
        .poll     =   sp_uio_of_##mber##_irq_poll,\
    };\
    struct miscdevice of_##mber##_irq_dev = \
    {\
        .minor    =   MISC_DYNAMIC_MINOR,\
        .fops     =   &of_##mber##_irq_fops,\
        .name     =   ""#mber"_irq",\
        .nodename =   "wr_fpga_ddr/"#mber"_irq",\
    };\

void clear_irq(u32 csr_base)
{
    WR_CSR_STATUS(csr_base, CSR_IRQ_SET_MASK);
}
static int sp_uio_open(struct inode *inode, struct file *filp)  
{  
    return 0;  
}  
static int sp_uio_release(struct inode *inode, struct file *filp)  
{  
    return 0;  
}

static ssize_t read_device_mem(u32 physical_addr, char* user_buf, size_t buf_count, loff_t* cur_offset)
{
    #define TMP_BUF_LEN 64
    static bool once_transfer = 1;
    int actual_len = 0;
    char result_buffer[TMP_BUF_LEN] = {0};

    sprintf(result_buffer, "%u\r\n", physical_addr);

    if (once_transfer == 0)
    {
       once_transfer = 1;
       return 0;
    }
    actual_len = strlen(result_buffer);
    if (copy_to_user(user_buf, result_buffer, actual_len))
    {
       return -EFAULT;
    }
    once_transfer = 0; 
    return actual_len;
}

static int mmap_device_mem(u32 physical_addr, struct vm_area_struct *vma)
{
    int ret;
    long length = vma->vm_end - vma->vm_start;
    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    vma->vm_flags |= (VM_IO | VM_LOCKED | (VM_DONTEXPAND | VM_DONTDUMP));
    ret = remap_pfn_range(vma, vma->vm_start, PFN_DOWN(physical_addr) + vma->vm_pgoff, length, vma->vm_page_prot);
    if (ret < 0) 
    {
        printk(KERN_ERR "mmap_device_mem: remap failed (%d)\n", ret);
        return ret;
    } 
    return 0;
}

typedef struct {
    u32 phy_write_csr;
    u32 kvir_write_csr;
    u32 phy_write_slave;
    u32 phy_read_csr;
    u32 kvir_read_csr;
    u32 phy_read_slave;
    u32 phy_read_arm_buffer;
    u32 phy_write_arm_buffer;
    int of_write_irq;
    int of_read_irq;
} wr_fpga_io_t;
