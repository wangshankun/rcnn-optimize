#include<linux/init.h>
#include<linux/kernel.h>
#include<linux/module.h>
#include<linux/slab.h>
#include<linux/delay.h>
#include<linux/of.h>  
#include<linux/sched.h>  
#include<linux/device.h>   
#include<linux/types.h>   
#include<linux/interrupt.h>
#include<linux/platform_device.h>
#include<asm/uaccess.h>
#include<linux/dma-mapping.h> 
#include"wr_fpga_ddr.h"

wr_fpga_io_t  wr_fpga_io =
{
    .of_write_irq    = -1,
    .of_read_irq     = -1,
};

MEM_OPS_API_DEFINE(write_csr)
MEM_OPS_API_DEFINE(write_slave)
MEM_OPS_API_DEFINE(read_csr)
MEM_OPS_API_DEFINE(read_slave)
MEM_OPS_API_DEFINE(write_arm_buffer)
MEM_OPS_API_DEFINE(read_arm_buffer)
IRQ_OPS_API_DEFINE(write)
IRQ_OPS_API_DEFINE(read)

static void *write_src_vir;
static void *read_dts_vir;

static int wr_fpga_ddr_probe(struct platform_device *pdev)
{
    struct resource *res = NULL;
    res = platform_get_resource_byname(pdev, IORESOURCE_MEM, "write_csr_regs");
    wr_fpga_io.kvir_write_csr = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(wr_fpga_io.kvir_write_csr))
    {
        return PTR_ERR(wr_fpga_io.kvir_write_csr);        
    }
    wr_fpga_io.phy_write_csr = res->start;
    res = platform_get_resource_byname(pdev, IORESOURCE_MEM, "read_csr_regs");
    wr_fpga_io.kvir_read_csr = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(wr_fpga_io.kvir_read_csr))
    {
        return PTR_ERR(wr_fpga_io.kvir_read_csr);        
    }
    wr_fpga_io.phy_read_csr = res->start;
    res = platform_get_resource_byname(pdev, IORESOURCE_MEM, "write_slave_regs");
    wr_fpga_io.phy_write_slave = res->start;
    res = platform_get_resource_byname(pdev, IORESOURCE_MEM, "read_slave_regs");
    wr_fpga_io.phy_read_slave = res->start;

    wr_fpga_io.of_write_irq = platform_get_irq_byname(pdev, "write_irq");
    if (wr_fpga_io.of_write_irq < 0)
    {
        printk("error platform_get_irq  wr_fpga_io.write_irq fail\n");
        return wr_fpga_io.of_write_irq;
    }
    wr_fpga_io.of_read_irq = platform_get_irq_byname(pdev, "read_irq");
    if (wr_fpga_io.of_read_irq < 0)
    {
        printk("error platform_get_irq  read_irq fail\n");
        return wr_fpga_io.of_read_irq;
    }
    if(request_irq(wr_fpga_io.of_write_irq, write_irq_fuc, 0, "w_fpga_ddof_read_irq", NULL) != 0)
    {
        printk("request_irq fail %s  %d\r\n",__FUNCTION__,__LINE__);
        return -1;
    }
    if(request_irq(wr_fpga_io.of_read_irq, read_irq_fuc, 0, "r_fpga_ddof_read_irq", NULL) != 0)
    {
        printk("request_irq fail %s  %d\r\n",__FUNCTION__,__LINE__);
        return -1;
    }

    write_src_vir  = dma_alloc_coherent(NULL, DMA_LEN, &(wr_fpga_io.phy_write_arm_buffer), GFP_KERNEL);
    read_dts_vir   = dma_alloc_coherent(NULL, DMA_LEN, &(wr_fpga_io.phy_read_arm_buffer), GFP_KERNEL);

    misc_register(&write_csr_dev);
    misc_register(&write_slave_dev);
    misc_register(&read_csr_dev);
    misc_register(&read_slave_dev);
    misc_register(&read_arm_buffer_dev);
    misc_register(&write_arm_buffer_dev);
    misc_register(&of_write_irq_dev);
    misc_register(&of_read_irq_dev);
    return 0;
}

static int wr_fpga_ddr_remove(struct platform_device *pdev)
{    
    misc_deregister(&write_csr_dev);
    misc_deregister(&write_slave_dev);
    misc_deregister(&read_csr_dev);
    misc_deregister(&read_slave_dev);
    misc_deregister(&read_arm_buffer_dev);
    misc_deregister(&write_arm_buffer_dev);
    misc_deregister(&of_write_irq_dev);
    misc_deregister(&of_read_irq_dev);

    free_irq(wr_fpga_io.of_write_irq, NULL);
    free_irq(wr_fpga_io.of_read_irq, NULL);
    
    dma_free_coherent(NULL, DMA_LEN, write_src_vir, wr_fpga_io.phy_write_arm_buffer);
    dma_free_coherent(NULL, DMA_LEN, read_dts_vir, wr_fpga_io.phy_read_arm_buffer);
	return 0;
}

static const struct of_device_id wr_fpga_ddr_0_of_match[] = {
    { .compatible = "altera,wr_fpga_ddr_0", },
    {},
};
MODULE_DEVICE_TABLE(of, wr_fpga_ddr_0_of_match);

static struct platform_driver wr_fpga_ddr_0_driver = {
    .probe  = wr_fpga_ddr_probe,
    .remove = wr_fpga_ddr_remove,
    .driver = {
            .name   = "wr_fpga_ddr_0",
            .of_match_table = of_match_ptr(wr_fpga_ddr_0_of_match),
    },
};

static int __init wr_fpga_ddr_0_init_driver(void)
{
    return platform_driver_register(&wr_fpga_ddr_0_driver);
}
subsys_initcall(wr_fpga_ddr_0_init_driver);

static void __exit wr_fpga_ddr_0_exit_driver(void)
{
    platform_driver_unregister(&wr_fpga_ddr_0_driver);
}
module_exit(wr_fpga_ddr_0_exit_driver);

MODULE_LICENSE("GPL");

