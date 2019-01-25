from logger import Logger


class viewSummary(object):
    
    def __init__(self, log_dir):
        from logger import Logger
        
        self.logger = Logger(log_dir)
        
    def view_stats(self, stats, epoch):
        for tag, value in stats.items():
            self.logger.scalar_summary(tag, value[0], epoch)
        
    def view_image(self, image_stat, epoch):
        for tag, image in image_stat.items():
                self.logger.image_summary(tag, image, epoch)
            
    def view_parameters(self, model):
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
            logger.histo_summary(tag+'/grad', value.grad.data.cpu(), epoch)
            
            
