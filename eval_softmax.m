function outputs = eval_softmax(net, imdb)

if strcmp(net.device, 'cpu')
    net.move('gpu');
end

net.conserveMemory = 0;
nsamp = 1;
outputs = {};


% outputs = [];
train = [] ;
% if isempty(train), train = find(imdb.images.set==1) ; end
% if isempty(train), train = find(imdb.images.set==1) ; end
% if isempty(train), train = find(imdb.images.set==1) ; end
if isempty(train), train = [1:size(imdb.images.set,2)] ; end

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



% while nsamp <= size(imdb.images.data, 4)
while nsamp <= size(train, 2)
%     step = min(256, size(imdb.images.data, 4) - nsamp+1);\

    step = min(128, size(train, 2) - nsamp+1);
    fprintf('%d out of %d to forward \n',nsamp, size(train, 2))
    
    batch = nsamp:1:nsamp+step-1;
    
    images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
    images = cnn_imagenet_get_batch(images,bopts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
    images = gpuArray(images) ;   
    
                          
    
%     images = gpuArray(single(imdb.images.data(:, :, :, nsamp:nsamp+step-1)));
%     inputs = {'image', images};
    inputs = {'data', images};
    net.eval(inputs) ;
    nsamp = nsamp + step;
    
    % Gather results.
	index = strfind({net.layers.name}, 'softmax'); %softmax
    index = find(not(cellfun('isempty', index)));
    npos = length(index);
    
    index2 = strfind({net.layers.name}, 'fc'); %softmax
    index2 = find(not(cellfun('isempty', index2)));
    nposFC = length(index2);

    if isempty(outputs)
        outputs = cell(1, nposFC);
    end
    
    lastPos = 1;
    for lix = 1:npos
        if ~strcmp(net.layers(index(lix)).name, 'softmax_global') && ~strcmp(net.layers(index(lix)).name, 'softmax_old')
            x = squeeze(gather(net.vars(net.layers(index(lix)).outputIndexes(1)).value));
            outputs{lastPos} = cat(2, outputs{lastPos}, x);
            lastPos = lastPos + 1;
        end
    end % lix
end

if strcmp(net.device, 'gpu')
    net.move('cpu');
end

