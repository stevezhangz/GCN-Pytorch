import torch
import tqdm
import os
from tensorboardX import SummaryWriter

def train(model,optimizer,epoches,feature,label,adj,loss,log,save_dir=None,validation=True,val_data=None):
    writer=SummaryWriter(comment='tensorboard_logging')
    if save_dir != None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        assert os.path.exists(save_dir)  , print("Wrong Path, Please update a new one")
    if val_data!=None:
        (val_feature,val_label)=val_data
    else:
        if validation:
            (val_feature, val_label)=(feature,label)
    progress_bar=tqdm.trange(epoches)
    total_len=feature.size()[0]
    log.info('Start trainning!')
    for epc in progress_bar:
        log.info('trainning stage···')
        model.train()
        optimizer.zero_grad()
        output=model(adj,feature)
        loss_val=loss(output,label.long())
        loss_val.backward()
        optimizer.step()
        _,top_5_logits=output.detach().cpu().topk(k=5,dim=1,largest=True,sorted=True)
        _,top_1_logits=output.detach().cpu().topk(k=1,dim=1,largest=True,sorted=True)

        correct_top1=torch.eq(top_1_logits.int(),label.view(-1,1).cpu().int()).sum()
        correct_top5=torch.eq(top_5_logits.int(),label.view(-1,1).cpu().int()).sum()

        top1_acc=correct_top1/total_len
        top5_acc=correct_top5/total_len
        progress_bar.set_postfix(Epoch=epc,Status='Train', top1_accuracy=top1_acc, top5_accuracy=top5_acc, loss=loss_val.detach().cpu().numpy())

        log.info(f'Train process: Training Loss:[{loss_val}]   Training Top1 acc:[{top1_acc}] Training Top5 acc:[{top5_acc}]   Epoch:[{epc}]')
        writer.add_scalar('train/loss',loss_val,epc)
        writer.add_scalar('train/acc1',top1_acc,epc)
        writer.add_scalar('train/acc5',top5_acc,epc)
        if validation:
            log.info('validation stage···')
            loss_val, top1_acc, top5_acc=eval(model,val_feature,val_label,adj,loss)
            progress_bar.set_postfix(Epoch=epc,Status='Val', top1_accuracy=top1_acc, top5_accuracy=top5_acc,
                                     loss=loss_val.detach().cpu().numpy())
            log.info(f'Eval stage····')
            log.info(f'Test process: Test Loss:[{loss_val}]   Test Top1 acc:[{top1_acc}] Test Top5 acc:[{top5_acc}]   Epoch:[{epc}]')
            writer.add_scalar('test/loss', loss_val, epc)
            writer.add_scalar('test/acc1', top1_acc, epc)
            writer.add_scalar('test/acc5', top5_acc, epc)

    if save_dir!=None: torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()
    },save_dir+'/model.pth')
    return loss_val, top1_acc, top5_acc


def eval(model,feature,label,adj,loss):
    model.eval()
    total_len=feature.size()[0]
    model.train()
    output = model( adj,feature)
    loss_val = loss(output, label.long())
    _, top_5_logits = output.detach().cpu().topk(k=5, dim=1, largest=True, sorted=True)
    _, top_1_logits = output.detach().cpu().topk(k=1, dim=1, largest=True, sorted=True)

    correct_top1 = torch.eq(top_1_logits.int(), label.view(-1, 1).cpu().int()).sum()
    correct_top5 = torch.eq(top_5_logits.int(), label.view(-1, 1).cpu().int()).sum()

    top1_acc = correct_top1 / total_len
    top5_acc = correct_top5 / total_len

    return loss_val,top1_acc,top5_acc








