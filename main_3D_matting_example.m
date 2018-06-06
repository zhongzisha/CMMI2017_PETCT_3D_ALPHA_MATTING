function main_3D_matting_example()
% by Zisha Zhong, email: zhongzisha@outlook.com
% @incollection{zhong_3d_2017,
% 	series = {Lecture {Notes} in {Computer} {Science}},
% 	title = {3D {Alpha} {Matting} {Based} {Co}-segmentation of {Tumors} on {PET}-{CT} {Images}},
% 	isbn = {978-3-319-67563-3 978-3-319-67564-0},
% 	url = {https://link.springer.com/chapter/10.1007/978-3-319-67564-0_4},
% 	language = {en},
% 	urldate = {2017-09-27},
% 	booktitle = {Molecular {Imaging}, {Reconstruction} and {Analysis} of {Moving} {Body} {Organs}, and {Stroke} {Imaging} and {Treatment}},
% 	publisher = {Springer, Cham},
% 	author = {Zhong, Zisha and Kim, Yusung and Buatti, John and Wu, Xiaodong},
% 	month = sep,
% 	year = {2017},
% 	doi = {10.1007/978-3-319-67564-0_4},
% 	pages = {31--42}
% }
% @article{levin_closed-form_2008,
% 	title = {A {Closed}-{Form} {Solution} to {Natural} {Image} {Matting}},
% 	volume = {30},
% 	issn = {0162-8828},
% 	doi = {10.1109/TPAMI.2007.1177},
% 	number = {2},
% 	journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
% 	author = {Levin, A. and Lischinski, D. and Weiss, Y.},
% 	month = feb,
% 	year = {2008},
% 	pages = {228--242}
% }

names={'A-IA0002109-1-0x0'};
fp = fopen('matting_time.txt','a');
for i=1:length(names)
    name = names{i};
    one_case_for_time(name, fp); 
end
fclose(fp);


end

function one_case_for_time(name, fp)
%% 1. do some preprocessing
%  2. do active contour
%  3. do image matting
%  4. do 3D image matting
%  5. save matting alpha for further processing

ctf = load_untouch_nii(['data/', name, '/InputCT_ROI.nii.gz']);
petf = load_untouch_nii(['data/', name, '/InputPET_SUV_ROI.nii.gz']);
gtf = load_untouch_nii(['data/', name, '/GTV_Primary_ROI_CT.nii.gz']);
obf = load_untouch_nii(['data/', name, '/ob.nii.gz']); % foreground seeds
bgf = load_untouch_nii(['data/', name, '/bg.nii.gz']); % background region
ct = single(ctf.img);
pet = single(petf.img);
gt = uint8(gtf.img);
ob = uint8(obf.img);
bg = uint8(bgf.img);
[H,W,C] = size(gt);

ct_min = -1024;
ct_max = 200.0;
ct(ct<ct_min) = ct_min;
ct(ct>ct_max) = ct_max;

ob = uint8((ob==1 & gt==1))*255;
bg = uint8((bg==1))*255;
gt = uint8((gt==1))*255;

[ct_ac, pet_ac, petct_ac_and, petct_ac_trimap1, ct_ac_trimap, pet_ac_trimap] = one_case_gen_trimap(pet, ct, ob, bg);

petct_ac_trimap1(ob>0) = 255;

tic;
[ct_3d_alpha, pet_3d_alpha] = do_3D_matting(pet, ct, ob, bg, petct_ac_trimap1);
matting_time = toc;

if ~isempty(fp)
    fprintf(fp, '%s,%d,%d,%d,%f\n',name,H,W,C,matting_time);
end
fprintf('%s,%d,%d,%d,%f\n',name,H,W,C,matting_time);


petct_ac_trimap1 = uint8(petct_ac_trimap1);
% save alpha as unary cost
petct_ac_trimap1_f = gtf;
petct_ac_trimap1_f.img = petct_ac_trimap1;
save_untouch_nii(petct_ac_trimap1_f, ['data/', name, '/petct_ac_trimap.nii.gz']);

ct_3d_matting_unary_cost = petf;
ct_3d_matting_unary_cost.img = single(ct_3d_alpha);
save_untouch_nii(ct_3d_matting_unary_cost, ['data/', name, '/ct_3d_matting_unary_cost.nii.gz']);
pet_3d_matting_unary_cost = petf;
pet_3d_matting_unary_cost.img = single(pet_3d_alpha);
save_untouch_nii(pet_3d_matting_unary_cost, ['data/', name, '/pet_3d_matting_unary_cost.nii.gz']);

end

function [ct_ac, pet_ac,petct_ac_and, petct_ac_trimap, ct_ac_trimap, pet_ac_trimap] = one_case_gen_trimap(pet, ct, ob, bg)

ct_ac = false(size(ct));
pet_ac = false(size(pet));
petct_ac_and = false(size(ct));
petct_ac_trimap = bg;
petct_ac_trimap(petct_ac_trimap==255) = 128;
ct_ac_trimap = bg;
ct_ac_trimap(ct_ac_trimap==255) = 128;
pet_ac_trimap = bg;
pet_ac_trimap(pet_ac_trimap==255) = 128;
se = strel('disk',1);
for i=1:size(pet,3)
    if ~isempty(find(ob(:,:,i)==255, 1))
        ct1 = ct(:,:,i);
        pt1 = pet(:,:,i);
        ob1 = ob(:,:,i)>0;
        if 1
            ob1_fill_holes = imfill(ob1, 'holes');
            ob1_holes = logical(ob1_fill_holes-ob1);
            ct1_median = median(ct1(ob1));
            ct1_min = max(-412, ct1_median - 0.9*ct1_median);%min(ct1(ob1));
            ct1_max = min(max(ct1(ob1)), ct1_median + 0.5*ct1_median);%median(ct1(ob1));
            
            pt1_min = min(pt1(ob1));
            pt1_max = mean(pt1(ob1));
            ct_ac0 = activecontour(ct1, ob1);
            pt_ac0 = activecontour(pt1, ob1);
            pos1 = find(ct_ac0>0);
            ct2 = ct1(pos1);
            ct_ac1 = ct_ac0;
            ct_ac1(pos1(ct2<ct1_min)) = 0;
            ct_ac1(pos1(ct2>ct1_max)) = 0;
            ct_ac1 = ct_ac1 | ob1;
            ct_ac2 = imerode(ct_ac1, se);
            ct_ac3 = imfill(ct_ac2, 'holes');
            ct_ac4 = imerode(ct_ac3, se);
            ct_ac5 = imdilate(ct_ac4, se);
            ct_ac5(ob1_holes) = 0;
            ct_ac(:,:,i) = ct_ac5 | ob1;
            if 0
                figure,
                subplot(3,4,1),imshow(mat2gray(ct1))
                subplot(3,4,2),imshow(mat2gray(pt1))
                subplot(3,4,3),imshow(gt(:,:,i))
                subplot(3,4,4),imshow(ob1)
                subplot(3,4,5),imshow(ct_ac0)
                subplot(3,4,6),imshow(pt_ac0)
                subplot(3,4,7),imshow(ct_ac1)
                subplot(3,4,8),imshow(ct_ac2)
                subplot(3,4,9),imshow(ct_ac3)
                subplot(3,4,10),imshow(ct_ac4)
                subplot(3,4,11),imshow(ct_ac5)
                subplot(3,4,12),imshow(ct_ac(:,:,i))
            end
            pet_ac(:,:,i) = pt_ac0;
        else 
            ct_ac(:,:,i) = activecontour(ct1, ob1);
            pet_ac(:,:,i) = activecontour(pt1, ob1);
        end
        petct_ac_and(:,:,i) = pet_ac(:,:,i) & ct_ac(:,:,i);
        % petct_ac_and(:,:,i) = imfill(petct_ac_and(:,:,i),'holes');
        % erode the object region
        % petct_ac_and(:,:,i) = imerode(petct_ac_and(:,:,i), se);
        trimap = petct_ac_trimap(:,:,i);
        if length(unique(trimap(:)))==1
            trimap(1:2,:) = 0;
            trimap(end-1:end,:) = 0;
            trimap(:, 1:2) = 0;
            trimap(:, end-1:end) = 0;
        end
        trimap(petct_ac_and(:,:,i)>0) = 255;
        petct_ac_trimap(:,:,i) = trimap; 
        
        % for ct_trimap
        trimap = ct_ac_trimap(:,:,i);
        if length(unique(trimap(:)))==1
            trimap(1:2,:) = 0;
            trimap(end-1:end,:) = 0;
            trimap(:, 1:2) = 0;
            trimap(:, end-1:end) = 0;
        end
        trimap(ct_ac(:,:,i)>0) = 255;
        ct_ac_trimap(:,:,i) = trimap; 
        
        
        trimap = pet_ac_trimap(:,:,i);
        if length(unique(trimap(:)))==1
            trimap(1:2,:) = 0;
            trimap(end-1:end,:) = 0;
            trimap(:, 1:2) = 0;
            trimap(:, end-1:end) = 0;
        end
        trimap(pet_ac(:,:,i)>0) = 255;
        pet_ac_trimap(:,:,i) = trimap; 
    end
end
end



function [ct_3d_alpha, pet_3d_alpha] = do_3D_matting(pet, ct, ob, bg, petct_ac_trimap)
[H,W,C] = size(ct);
ct_3d_alpha = zeros(size(ct));
pet_3d_alpha = zeros(size(ct));
fprintf('do 3D matting ...\n');

if nargin==4 || ~exist('petct_ac_trimap','var') || isempty(petct_ac_trimap)
    petct_ac_trimap = zeros(size(ct));
    petct_ac_trimap(ob>0) = 255;
    petct_ac_trimap(bg>0 & ob==0) = 128; 
    if isempty(find(bg==0, 1))
        petct_ac_trimap(1:5,:, :) = 0;
        petct_ac_trimap(end-5:end,:, :) = 0;
        petct_ac_trimap(:, 1:5, :) = 0;
        petct_ac_trimap(:, end-5:end, :) = 0;
        petct_ac_trimap(:, :, 1:5) = 0;
        petct_ac_trimap(:, :, end-5:end) = 0;
    end
end
consts_map = logical(petct_ac_trimap==0 | petct_ac_trimap==255);
consts_vals = double(petct_ac_trimap==255);

epsilon = 1e-7;
win_size = 1;
ct = double(ct);
pet = double(pet);

fprintf('size: %d\n', numel(ct));
bg_bw = bg>0;
ob_bw = ob>0;

regions = regionprops(bg_bw, 'BoundingBox');
for ri=1:length(regions)
    w1 = regions(ri).BoundingBox(1);
    h1 = regions(ri).BoundingBox(2);
    c1 = regions(ri).BoundingBox(3);
    w2 = regions(ri).BoundingBox(4);
    h2 = regions(ri).BoundingBox(5);
    c2 = regions(ri).BoundingBox(6);
    h1 = max(1, floor(h1));
    w1 = max(1, floor(w1));
    c1 = max(1, floor(c1));
    h2 = min(H, h1+h2);
    w2 = min(W, w1+w2);
    c2 = min(C, c1+c2);
    hh = h1:h2;
    ww = w1:w2;
    cc = c1:c2;
    if length(hh)*length(ww)*length(cc)>5e5
        ct1 = ct(hh, ww, cc);
        pet1 = pet(hh, ww, cc);
        consts_map1 = consts_map(hh, ww, cc);
        consts_vals1 = consts_vals(hh, ww, cc);
        [ct1_H, ct1_W, ct1_C] = size(ct1);
        % downsample the data and get alpha quickly
        % maybe have bug here, due to no object seeds after downsampling
        ct1 = imresize(ct1, [floor(ct1_H/2) floor(ct1_W/2)], 'nearest');
        pet1 = imresize(pet1, [floor(ct1_H/2) floor(ct1_W/2)], 'nearest');
        consts_map1 = imresize(consts_map1, [floor(ct1_H/2) floor(ct1_W/2)], 'nearest');
        consts_vals1 = imresize(consts_vals1, [floor(ct1_H/2) floor(ct1_W/2)], 'nearest');
        ct_3d_alpha1=solveAlpha(ct1,consts_map1,consts_vals1,epsilon,win_size);
        pet_3d_alpha1=solveAlpha(pet1,consts_map1,consts_vals1,epsilon,win_size);
        % upsample the alpha
        ct_3d_alpha1 = imresize(ct_3d_alpha1, [ct1_H ct1_W], 'nearest');
        pet_3d_alpha1 = imresize(pet_3d_alpha1, [ct1_H ct1_W], 'nearest');
    else
        ct1 = ct(hh, ww, cc);
        pet1 = pet(hh, ww, cc);
        consts_map1 = consts_map(hh, ww, cc);
        consts_vals1 = consts_vals(hh, ww, cc);
        ct_3d_alpha1=solveAlpha(ct1,consts_map1,consts_vals1,epsilon,win_size);
        pet_3d_alpha1=solveAlpha(pet1,consts_map1,consts_vals1,epsilon,win_size);
    end
    ct_3d_alpha(hh, ww, cc) = max(min(ct_3d_alpha1,1),0);
    pet_3d_alpha(hh, ww, cc) = max(min(pet_3d_alpha1,1),0);
end
end

function alpha=solveAlpha(V,consts_map,consts_vals,epsilon,win_size)

[h,w,c]=size(V);
img_size=w*h*c;

A=getLaplacian1(V,consts_map,epsilon,win_size);

D=spdiags(consts_map(:),0,img_size,img_size);
lambda=1000;
% x=(A+lambda*D)\(lambda*consts_map(:).*consts_vals(:));
% x=pcg(A+lambda*D, lambda*consts_map(:).*consts_vals(:));
x=pcg(A+lambda*D, lambda*consts_map(:).*consts_vals(:), 1e-6, 300);

alpha=max(min(reshape(x,h,w,c),1),0);
end

function [A,A1]=getLaplacian1(V,consts,epsilon,win_size)

if (~exist('epsilon','var'))
    epsilon=0.0000001;
end
if (isempty(epsilon))
    epsilon=0.0000001;
end
if (~exist('win_size','var'))
    win_size=1;
end
if (isempty(win_size))
    win_size=1;
end

neb_size=(win_size*2+1)^3;
[h,w,c]=size(V);
img_size=w*h*c;
% consts=imerode(consts,ones(win_size*2+1));strel('cube',3)
% consts=imerode(consts,strel('cube',3));

indsM=reshape([1:img_size],h,w,c);

tlen=sum(sum(sum(1-consts(win_size+1:end-win_size,win_size+1:end-win_size,win_size+1:end-win_size))))*(neb_size^2);

row_inds=zeros(tlen ,1);
col_inds=zeros(tlen,1);
vals=zeros(tlen,1);
len=0;
for k=1+win_size:c-win_size
    for j=1+win_size:w-win_size
        for i=win_size+1:h-win_size
            if (consts(i,j,k))
                continue
            end
            win_inds=indsM(i-win_size:i+win_size,j-win_size:j+win_size, k-win_size:k+win_size);
            win_inds=win_inds(:);
            winI=V(i-win_size:i+win_size,j-win_size:j+win_size,k-win_size:k+win_size);
            winI=reshape(winI,neb_size,1);
            win_mu=mean(winI,1)';
            win_var=inv(winI'*winI/neb_size-win_mu*win_mu' +epsilon/neb_size);
            
            winI=winI-repmat(win_mu',neb_size,1);
            tvals=(1+winI*win_var*winI')/neb_size;
            
            row_inds(1+len:neb_size^2+len)=reshape(repmat(win_inds,1,neb_size),...
                neb_size^2,1);
            col_inds(1+len:neb_size^2+len)=reshape(repmat(win_inds',neb_size,1),...
                neb_size^2,1);
            vals(1+len:neb_size^2+len)=tvals(:);
            len=len+neb_size^2;
        end
    end
end

vals=vals(1:len);
row_inds=row_inds(1:len);
col_inds=col_inds(1:len);
A=sparse(row_inds,col_inds,vals,img_size,img_size);

sumA=sum(A,2);
A=spdiags(sumA(:),0,img_size,img_size)-A;

end
