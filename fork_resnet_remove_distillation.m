function [net, derOutputs] = fork_resnet_remove_distillation(net)
opts.orig_loss='for_keep';
%% Update loss layer for the old layer. Only the "last new" task is updated.
if strcmp(opts.orig_loss, 'for_keep')
% if false
    index = strfind({net.layers.name}, 'loss_distillation');
    index = find(not(cellfun('isempty', index)));
    
    net.removeLayer(net.layers(index(1)).name);
    % derOutputs = derOutputs(1:end-2);
    derOutputs = opts.derOutputs(1:end-2);
end



