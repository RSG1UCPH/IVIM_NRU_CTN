%% --- Processes and fits DWIs to IVIM-model using AFNI.
%  --- Performs fitting to IVIM-model using least squared curves method.

%% --- Define input and set environment:
addpath('/data2/KristofferBrix/IVIM/processingPipeline/');
inDir = '/data2/KristofferBrix/IVIM/processingPipeline/rawdata';
outDir = '/data2/KristofferBrix/IVIM/processingPipeline/proc';

if isfile(outDir) == 0
    mkdir(outDir);
end

%% --- Fetch and convert DWI trace DICOMs:
dwis = regexp(split(ls(fullfile(inDir))),...
    'EP2D_DIFF_4SCAN_TRACE_P2_MONO_FULL_COVERAGE_TRACEW_\w*','match');
dwis = dwis(~cellfun(@isempty, dwis));

for i = 1:length(dwis)
    unix(['dcm2niix -f dwi_tra_' num2str(i) ' ' fullfile(inDir,char(dwis{i}))]);
    movefile(fullfile(inDir,char(dwis{i}),['dwi_tra_' num2str(i) '*']), outDir);

%% --- Motion correction with AFNI:
    unix(['3dvolreg'...
        ' -prefix ' fullfile(outDir,['dwi_moco_' num2str(i)]) ...
        ' -dfile ' fullfile(outDir,['motionParams_' num2str(i) '.txt '])...
        fullfile(outDir,['dwi_tra_' num2str(i) '.nii'])]);
    unix(['3dAFNItoNIFTI'...
        ' -prefix ' fullfile(outDir,['m_dwi_' num2str(i)]) ...
        ' ' fullfile(outDir,['dwi_moco_' num2str(i) '+orig'])]);
%% --- Brain extraction:
    unix(['bet'...
        ' ' fullfile(outDir,['m_dwi_' num2str(i)]) ...
        ' ' fullfile(outDir,['bet_' num2str(i)]) ' -m']);
    unix(['fslmaths'...
        ' ' fullfile(outDir,['m_dwi_' num2str(i) '.nii']) ...
        ' -mas ' fullfile(outDir,['bet_' num2str(i) '_mask.nii.gz']) ...
        ' ' fullfile(outDir,['bm_dwi_' num2str(i)])]);

%% --- Prepare for fit:
    b = split(fileread(fullfile(outDir,['dwi_tra_' num2str(i) '.bval'])));
    b = b(~cellfun(@isempty, b));
    b = cell2mat(cellfun(@(x) str2num(x),b,'UniformOutput',false));
    
    V = niftiread(fullfile(outDir,['bm_dwi_' num2str(i) '.nii.gz']));
    voxels = length(V(:,1,1,1))*length(V(1,:,1,1))*length(V(1,1,:,1));
    disp('Fitting data to bi-exponential decay model.');
    parfor_progress(voxels);
    
    tic,
    
%% --- Fitting data to bi-exponential decay model: 
    writematrix([],'temp.txt');
    parfor (j = 1:voxels,8)
        try
            opts = optimset('Display','off');
            [x, y, z] = ind2sub(size(V), j);
            S = squeeze(double(V(x,y,z,:)));        
            ivim_params = [0 0 0 0 0 0 0];

            if sum(S) == 0
                [x, y, z] = ind2sub(size(V), j);
                ivim_params(5:7) = [x y z];
                writematrix(ivim_params,'temp.txt','WriteMode','append');
            else
                %%% --- Single exponential fit on b > 200 for estimation of D:
                fun1 = @(p,b)p(1)*exp(-p(2)*b);                
                ivim_params(1:2) = lsqcurvefit(fun1,[1000 0.0001],b(4:5),S(4:5),[0 0],[2000 0.001],opts);

                %%% --- Extrapolate monoexponential fit for calculating f:
                ivim_params(3) = (S(1)-ivim_params(1))/(S(1));

                %%% --- Bi-exponential fit on all b-values for
                %%% estimation of remaining IVIM-parameters:
                fun2 = @(Dstar,b)S(1)*(ivim_params(3)*exp(-ivim_params(2)*b)+(1-ivim_params(3))*(exp(-Dstar*b)));
                ivim_params(4) = lsqcurvefit(fun2,[0.001],b,S,[0],[0.1],opts);

                %%% --- Estimated predicted S(0):
                fun3 = @(S,b)S(1)*(ivim_params(3)*exp(-ivim_params(2)*b)+(1-ivim_params(3))*(exp(-ivim_params(4)*b)));
                ivim_params(1) = lsqcurvefit(fun3,[1000],b,S,[0],[2000],opts);
                
                [x, y, z] = ind2sub(size(V), j);
                ivim_params(5:7) = [x y z];
                writematrix(ivim_params,'temp.txt','WriteMode','append');
            end
        catch
            [x, y, z] = ind2sub(size(V), j);
            ivim_params(5:7) = [x y z];
            writematrix(ivim_params,'temp.txt','WriteMode','append');
        end
        parfor_progress;
    end
    parfor_progress(0);
    toc,
    poolobj = gcp('nocreate');
    delete(poolobj);

 %% --- Writes IVIM maps:
    disp('Writing files...');
    ivim_params = readmatrix('temp.txt');
    nonPerf_S0 = V;
    D = V;
    f = V;
    Dstar = V;
    for j = 1:voxels
        voxel = ivim_params(j,:);
        x = voxel(5);
        y = voxel(6);
        z = voxel(7);
        nonPerf_S0(x,y,z) = voxel(1);
        D(x,y,z) = voxel(2);
        f(x,y,z) = voxel(3);
        Dstar(x,y,z) = voxel(4);
    end
    
    I = niftiinfo(fullfile(outDir,['bm_dwi_' num2str(i) '.nii.gz']));
    niftiwrite(D,fullfile(outDir,['D_' num2str(i)]),I);
    niftiwrite(f,fullfile(outDir,['f_' num2str(i)]),I);
    niftiwrite(Dstar,fullfile(outDir,['Dstar_' num2str(i)]),I);
    niftiwrite(nonPerf_S0,fullfile(outDir,['noPerf_S0_' num2str(i)]),I);
    disp('Fitting complete.');
    
    
    %% --- Gaussian smoothing with 0.75 mm kernel (half voxel width):
    IVIMmaps = {fullfile(outDir,['D_' num2str(i)]); ...
        fullfile(outDir,['f_' num2str(i)]); ...
        fullfile(outDir,['Dstar_' num2str(i)]); ...
        fullfile(outDir,['noPerf_S0_' num2str(i)])};    
    for j = 1:4
        map = char(IVIMmaps(j,:));
        unix(['fslmaths ' ...
            map ...
            ' -s 0.75 '...
            [map(1:end-1) 'smoothed_' num2str(i)]]);
    end
    
    %% --- Groups and cleans output:
    mkdir(fullfile(outDir,'IVIM'));
    movefile(fullfile(outDir,'f_*'),fullfile(outDir,'IVIM'));
    movefile(fullfile(outDir,'D_*'),fullfile(outDir,'IVIM'));
    movefile(fullfile(outDir,'Dstar_*'),fullfile(outDir,'IVIM'));
    movefile(fullfile(outDir,'noPerf_S0_*'),fullfile(outDir,'IVIM'));
    delete('temp.txt');
    delete(fullfile(outDir,'bet_*.nii.gz'));
end
