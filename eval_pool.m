function outputs = eval_pool(net, imdb, pos)

if strcmp(net.device, 'cpu')
    net.move('gpu');
end

net.conserveMemory = 0;
nsamp = 1;
outputs = [];
train = [] ;
if isempty(train), train = find(imdb.images.set==1) ; end

train = train(pos);

meta = net.meta;
opts.numFetchThreads = 12 ;
opts.numAugments = 1 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
bopts.transformation = meta.augmentation.transformation ;
bopts.numAugments = opts.numAugments ; 


%  fprintf('%d to forward \n', size(train, 2));
while nsamp <= size(train, 2)
    step = min(128, size(train, 2) - nsamp+1);
    
    fprintf('%d out of %d to forward \n',nsamp, size(train, 2))
%     tic
    batch = nsamp:1:nsamp+step-1;
    
    
    images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
    images = cnn_imagenet_get_batch(images,bopts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
    images = gpuArray(images) ;
%     toc;
%     tic;
% inputs = {'data', im, 'label', labels} ;
    
%     images = gpuArray(imdb.images.data(:, :, :, nsamp:nsamp+step-1));
    inputs = {'data', images};
    net.eval(inputs) ;
    nsamp = nsamp + step;
    
    % Gather results.
    index = strfind({net.layers.name}, 'pool_final');
    index = find(not(cellfun('isempty', index)));
    
    % Concat results.
    x = squeeze(gather(net.vars(net.layers(index(1)).outputIndexes(1)).value));
    outputs = cat(2, outputs, x);
%     toc;
end

if strcmp(net.device, 'gpu')
    net.move('cpu');
end


% 
% % -------------------------------------------------------------------------
% function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% % -------------------------------------------------------------------------
% images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
% isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;
% 
% if ~isVal
%   % training
%   im = cnn_imagenet_get_batch(images, opts, ...
%                               'prefetch', nargout == 0) ;
% else
%   % validation: disable data augmentation
%   im = cnn_imagenet_get_batch(images, opts, ...
%                               'prefetch', nargout == 0, ...
%                               'transformation', 'none') ;
% end
% 
% if nargout > 0
%   if useGpu
%     im = gpuArray(im) ;
%   end
%   labels = imdb.images.label(batch) ;
%   inputs = {'data', im, 'label', labels} ;
% end
