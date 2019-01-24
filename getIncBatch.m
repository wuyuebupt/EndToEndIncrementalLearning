%------------------------------------------------------------------------------------
% INTERNAL FUNCTIONS
%------------------------------------------------------------------------------------
% -------------------------------------------------------------------------
function inputs = getIncBatch(imdb, batch)
% -------------------------------------------------------------------------
% images = imdb.images.data(:,:,:,batch);
% labels = imdb.images.labels(batch) ;
labels = imdb.images.label(batch) ;


% opts.numFetchThreads = 12 ;
opts.numAugments = 1 ;

% bopts.numThreads = opts.numFetchThreads ;
bopts.numAugments = opts.numAugments ; 


bopts.imageSize = [224,224,3];
bopts.border = [32,32] ;
bopts.averageImage =   [121.0762;  113.2553;   98.7793] ;
bopts.rgbVariance = [-0.726353049278259,1.461229681968689,-0.755036294460297;-2.525028705596924,-0.007646363228559,2.414307355880737;6.486166477203369,6.740384101867676,6.804973125457764] ;
bopts.transformation = 'stretch';


images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
images = cnn_imagenet_get_batch(images,bopts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;

images = gpuArray(images) ;

% if ~isempty(imdb.opts.gpus)
%     images = gpuArray(images) ;
% end

j = 1;
jj = 3;
inputs = cell(1, length(imdb.meta.inputs) * 2);
for i=1:length(imdb.meta.inputs)-1
%     if strcmp(imdb.meta.inputs{i}, 'image')
%         inputs(1:2) = {'image', images};
    if strcmp(imdb.meta.inputs{i}, 'data')
        inputs(1:2) = {'data', images};
    else
        inputs(jj:jj+1) = {imdb.meta.inputs{i}, imdb.images.distillationLabels{j}(:, batch)};
        j = j + 1;
        jj = jj + 2;
    end
end

inputs(jj:jj+1) = {imdb.meta.inputs{end}, labels};


end