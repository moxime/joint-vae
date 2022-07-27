
def model_ensembling(*losses, operation='log_mean_exp'):
r"""model ensmbling

-- operation: log_mean_exp, mean, join, voting

"""
    if operation == 'log_mean_exp':

        t = torch.stack(losses)

        tref = t.max(axis=0)[0]
        dt = t - tref

        return (dt.exp().mean(axis=0).log() + tref).squeeze(0)

    if operation == 'mean':

        return torch.stack(losses).mean(0)

    if operation == 'join':

        
