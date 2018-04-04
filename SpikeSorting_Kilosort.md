#  Spike Sorting Using Kilosort
[Kilosort from "Cortex Lab"](https://github.com/cortex-lab/KiloSort)

For original publications:
[Pachitaru, 2016. NIPS.](https://papers.nips.cc/paper/6326-fast-and-accurate-spike-sorting-of-high-channel-count-probes-with-kilosort)
[Pachitaru, 2016. Bioarxiv preprint.](https://www.biorxiv.org/content/early/2016/06/30/061481)

# Downloading KiloSort
Navigate wherever you would like the KiloSort sourcecode to live and clone it to your computer. I'd suggest using `git`, so that if the repository is updated all you have to do is run `git pull` to get the updated version. But you can also just download the archive file.

```shell
cd path/to/your/favorite/source/code/location
git clone https://github.com/cortex-lab/KiloSort.git
```

[Here](https://github.com/cortex-lab/KiloSort/tree/master/Docs) are some instructions provided with the repository. They were helpful (especially for Windows which I hardly ever use), but not exhaustive.

*n.b., It appears this code was originally written for use with CUDA 7.5, whereas the current version of CUDA is 9.x. As you will see later, MATLAB 2018's default version of CUDA it looks for is 9.0. I am not sure (a) if Marius's CUDA code will work with version 9.x or (b) if MATLAB 2018 will work with CUDA 7.5.*

**UPDATE :** Using version 9.1 of the CUDA toolkit (rather than 7.5 or 9.0) did indeed work. Follow the below instructions.
**UPDATE :** MATLAB 2017 appears to be compatible with CUDA v8.x (see Windows-specific sections). It may still be possible to use CUDA v9.x with a deep dive into configuration files, but I decided not to fight that battle and downloaded CUDA v8.0 from the legacy downloads page.

# Downloading npy-matlab
To integrate with Phy (the curating viewer written in Python), KiloSort (which is Matlab-based) needs to be able to read/write numpy `*.npy` files. Apparently, the lab has written some code for this which you'll need to add to your path before you can run KiloSort/save the results.

```shell
cd path/to/your/favorite/source/code/location
git clone https://github.com/kwikteam/npy-matlab.git
```

# Installing CUDA toolkit
This is reasonably well-documented ([linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), [windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html), [macOS])(http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html).

Download the CUDA toolkit [here](https://developer.nvidia.com/cuda-downloads) & install. First, it would be a good idea to see what version of CUDA your version of MATLAB supports. Inspect the top of the configuration file(s) buried in the following location.
```shell
/path/to/MATLAB/R201xx/toolbox/distcomp/gpu/extern/src/mex/
```
*It appears that MATLAB 2018 supports CUDA v9, 2017 supports v8, and so on.*

**The Linux machine used for this installation has version 9.1, so I am going to see if KiloSort + MATLAB 2018 + CUDA 9.1 play nice together.** ~~This may crash and burn.~~

**I briefly tried using 9.1 with MATLAB 2016 and 2017 on the Windows machine used for this installation, but found this to be more work than it was worth. I instead dowloaded v8.0 for MATLAB 2017b on this machine.**

## Windows-specific Steps
### Installing Visual C++ Compiler
If using Windows, you will need to download+install Microsoft's Visual Studio (VS) in order to have a CUDA-compatible C++ compiler. The version of VS you download+install should match the version being searched for in the `mexcuda` configuration file (see `mexcuda` section below). For example, in MATLAB 2017b, inspecting the configuration file `nvcc_msvcpp2015.xml` it seems that the compiler it is looking for is `MSVCPP14` (Microsoft Visual C++ 14.0), which comes with VS 2015. So, VS 2015 is the version to download and install. *n.b.*, I wasted a lot of time trying to figure out why installing VS 2015 was not actually installing the compiler--or at least not installing it where MATLAB knew to look. It turns out that installing this compiler is a non-default option (see the [warning on this page](https://msdn.microsoft.com/en-us/library/60k1461a.aspx)). **You need to do the custom install and make sure the C/C++ options are checked.**

Once the C++ compiler is installed, you need to configure `mex` to actually use it. The easiest way to do this is to run mex with the setup option and let it give you the appropriate suggestions. In MATLAB, type the following.
```matlab
mex -setup
```
This should print to the console a number of options. MS Visual C++ 2015 (or whichever version you needed for your release of MATLAB/CUDA) should be one of the options now! If not, something went wrong with the previous step: *e.g.*, the C++ compiler option was not checked at install. Or whichever version you installed. Run the `mex -setup:` command, with the appriate string for the configuration file suggested by MATLAB.

### Prevent Windows from killing KiloSort
Also, it appears that Windows will kill a process (like KiloSort) running on a GPU too long if that card is also being used to render your display. First, open up a command window and launch the registry editor.
```
regedit
```

Navigate to `HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers`. Create new DWORD key called `TdrLevel`, and set its value to `0`.
**Disclaimer:**  I do not know the details of what this actually does. This comes from the KiloSort/Docs folder and I blindly obeyed.

### Setting CUDA Max Cache Size
Finally, the creator suggests setting the max cache size allowed for CUDA to something large like 1GB. To do this, create an environment variable (My PC -> Properties -> Advanced System Settings -> Environment Variables) with some large value like 1e9: `CUDA_CACHE_MAXSIZE` to `1073741824`. Note that I tried it without this step first and then again with it and I didn't notice any big difference, so this step may be unnecessary.

# Matlab `mexcuda` & Kilosort
KiloSort's GPU distributed computing is done by compiling CUDA code and calling from MATLAB. This functionality is built with MATLAB `mexcuda`, and this is the only "installation" step that is required before actually running KiloSort. After that, you just need to make sure that the code repository is in your MATLAB path and make sure you hand it a `struct` that contains the configuration options for sorting your dataset.

*n.b., At this writing I am using Matlab R2018a on a machine running Ubuntu 16.04 and outfitted with an NVIDIA GTX 1080 and 2 4-core CPUs. Some details may vary for other setups. This is a computer I only have access to temporarily while home for Easter!*

## Install Matlab's Parallell Computing Toolbox
If you haven't already, download/install the parallel computing toolbox.

## Give it a go as-is
As a first check, let's try running the author's `mexGPUall.m` file and see if it magically works. It won't! This is the file that we will be running to actually build the GPU functions that will be called later by KiloSort when it runs in MATLAB.

```shell
cd /path/to/KiloSort/CUDA

matlab -nodesktop #or you can just launch matlab with the GUI

mexGPUall
```

**Initial Problems (Linux)**
1. *"Warning: Version 9.0 of the CUDA toolkit could not be found. If installed, set MW_NVCC_PATH environment variable to location of nvcc compiler."*
2. *Error using mex
No supported compiler was found. For options, visit https://www.mathworks.com/support/compilers.*

Note that I found it was a little misleading  that Problem 1 was a non-fatal warning. I thought there was some other required compiler that was not available to mex that ultimated killed the process. However, after a few minutes of checks I found that MATLAB could indeed "see" `gcc`/`g++`. It really ended up being because of the first warning.

**Initial Problems (Windows)**
1. MATLAB could not find a CUDA-compatible C++ compiler (even after trying to install MSVC++ compiler). Following the steps above for the custom install of the appropriate version of Visual Studio fixed this problem.

## `mexcuda` configuration (Linux)
Problem 1 arose because both CUDA 8 and 9.1 were installed (rather than 9.0, which MATLAB was looking for) and the `nvcc` that was sym-linked into the existing `PATH` variable was from version 8. I set the environment variable as suggested by the warning and this did not immediately solve the problem. But it was a necessary step.

This is personal preference, but my M.O. is to edit/create environment variables using the `.profile` file located in the user home directory. I believe is one of the last files `bash` looks for at login. Some folks use `.bashrc` , `.bash_profile`, or other such files. There are some slight differences in how they're handled but in general it shouldn't matter.
```shell
nano ~/.profile
```
In the file of your choosing add the following line.
```shell
export MW_NVCC_PATH="/usr/local/cuda/bin"

# n.b. if you have multiple versions of CUDA installed, version 9.0 is at least the default for Matlab 2018.
# I used this instead:
export MW_NVCC_PATH="/usr/local/cuda-9.1/bin"
```

Presumably you (or the installer) have already added `CUDA_HOME` and added it to your `PATH` environment variable when you installed CUDA originally. If not, you'll want to do this. *Logout* of the current shell and login to a new shell (*i.e.*, just open up a new terminal). Confirm that the environment variable was made and exported and that the version of `nvcc` is 9.1.
```shell
echo $MW_NVCC_PATH
$MW_NVCC_PATH/nvcc --version
```
You will notice that the version of CUDA that was already installed is **9.1**, not version **9.0** that MATLAB warned us she wanted. Yes, I said "she". Clarissa insisted MATLAB was a "she". Maybe because you love her but she can be a ... cruel mistress.

As an experiment, I tried to simply override the version of the CUDA compiler (`nvcc`) that MATLAB was looking for. Buried in MATLAB's source code, there is an `.xml` file that has build configuration/instructions for mexGPU. I haven't looked into the details, but it seems like it is MATLAB mex's version of a GNU makefile.

**Warning**
1. Before editing this file, make a copy of the original in case something is broken in the process. **A safer alternative is to copy the contents of this file and overwrite the contents of the generic linux (or Windows) `*.xml` file into the `KiloSort/CUDA` subdirectory, and then do the edits suggested below.**
2. I am only trying this manual override because I assume the differences between version 9.0 and 9.1 are not that great. I would **not** suggest doing this if your version of MATLAB is looking for 7.x or 8.x.
**Update:** I tried using CUDA 9.x when MATLAB was looking for 8.0 with the Windows install, and found it was more work than it was worth.

```shell
cd /path/to/MATLAB/R2018a/toolbox/distcomp/gpu/extern/src/mex/glnxa64
sudo nano nvcc_g++.xml
```

You don't have to use `nano`, but I find it to be the most convenient commandline text editor. You can use any text editor. The key is that you have to have superuser priveleges in order to overwrite this file where it is in the default Linux installs. Again, it will be safer to simply copy the contents of this file into the appropriate `*.xml` file in the KiloSort repo.

In the file, I replaced any mention of "9.0" with "9.1". I believe there are 3-4 lines that need changing. Like I said this file (`nvcc_g++.xml`) appears to be a build/compiler configuration file. It is hard-coded to check for 9.0. Operating under the assumption the differences between 9.0 and 9.1 are small, I figured if I just changed the checks that are being done at configuration, things would run smoothly enough.

After doing that, try running `mexGPUall.m` again.

```shell
cd /path/to/KiloSort/CUDA

matlab -nodesktop #or you can just launch matlab with the GUI

mexGPUall
```

This "worked", but there are **a lot** of warnings that there are variables that were defined but not used and there was a seperate set of warnings regarding "narrowing" of ints. *e.g.*,

```shell
warning: narrowing conversion of ‘blocksPerGrid’ from ‘int’ to ‘mwSize {aka long unsigned int}’ inside { } [-Wnarrowing]
```
At least in C/C++ (but I assume it's also true using CUDA)  `int` is 16-bits. A `long int` is 32-bit, so I really don't understand how/why this would be a "narrowing" type conversion. But I may be misunderstanding the warning. I thought it was letting us know that there may be problems with overflow or something if it was going from higher to lower bit depth.

Anyway, it does succesfully complile now, but I am unsure of the resulting code is functional. I will be checking that next.

## `mexcuda` configuration (Windows)
As noted above, I opted to match versions of MSVC++ and CUDA (nvcc) to the release of MATLAB. I found it more difficult to *full-version* disparities in any one of these three players than it was worth. In particular, while mex would work with the latest version of MSVC++, I couldn't make the right changes to the `mexcuda` configuration file in 2017b to make this latest version work with CUDA.

Because versions of the C++ compiler and CUDA were what the MATLAB release expected, this configuration was simpler at the end of the day. I simply replaced the contents of `KiloSort/CUDA/mex_CUDA_win64.xml` with the contents of MATLAB's `nvcc_msvcpp2015.xml` configuration file buried in the usual location. Note the exact name of that file will vary depending on the version of Visual C++.

Once you've done that, simply re-run the `mexGPUall.m` script.

# Testing build on synthetic data
The KiloSort directory has an `eMouse` subdirectory. You will need to change some paths in `master_eMouse.m` for it to work for your computer. For example, you need to provide the path to KiloSort source code and the nmpy-matlab directory. Once you do that, simply run `master_eMouse.m`.

This file creates a 1000-second 32-channel synthetic dataset (fs=25kHz). In the comments, the creator notes that using a GTX 1080 GPU and an SSD, it took 55 seconds on synthetic data. Using the setup I currently have access to (which has the same GPU), I am seeing 38 seconds to completion. So, it is possible that we could see sorting times of 5% the time of the dataset. However, I imagine as the dataset gets larger, the efficiency of memory handling is probably going to keep this from happening on full datasets.

# Testing on Small Experimental Dataset
The Linux machine above finished loading, pre-processing, sorting, and saving results for a 4GB dataset (fs = 25 kHz) in 100 seconds.

The communal Windows spike sorting machine took 400 seconds. It took ~25 minutes to sort the entire 17 GB dataset that the original 4 GB dataset was sub-selected from.

# Data Pre-processing
## Formatting Data for Sorting
I could not find dedicated documentation for how one should format the binary data fed to KiloSort. Because of that, I am assuming it is the same as it was for Klusta. However, I did find the [eMouse simulation code](https://github.com/cortex-lab/KiloSort/blob/master/eMouse/make_eMouseData.m) for synthetic test data.

Going off of this file, it appears that
1. 16-bit signed integers (`Int16`, `int16`) is the data type required.
2. You will want to scale the single/double-precision floats before this conversion to `Int16` in order to make best use of the dynamic range of 16 bits. **Save the scaling factor in a seperate file!** Currently, I am using a factor of `scale = 5e5`, which maps 1 µV (`1e-6`) double-precision float to an integer value of 1. And it doesn't saturate until after tens of mV, so this should be no problem for the range of voltages for extracellular spiking data.
3.  The binary file needs to be multiplexed such that there is a timepoint for each channel in order before proceeding to the next point in time. Therefore, save the data oriented as `[nChan x nTime]`. Matlab's `fwrite()`  writes the data of a matrix in column order, so this will yield the desired binary file.

## Raw tdt -> mat format
My first pre-processing stage (regardless of doing spike sorting) is always to convert TDT files (collection of `*.sev`, `*.tev`, `*.tsq` files) into a `*.mat`-formatted HDF5 file (v7.3) that I can read/write in Matlab as well as easily read in C/C++, Python, or Julia, given an HDF5 library.

I will not go into details of the anatomy of this `*.mat` file here, but I will point out where I have saved the raw (*i.e.* unfiltered) extracellular voltages.

```
/data
    /streams
        /Vraw
            /data : *Array{Float32,2}(nTime, nChannel)*
            /fs : *Float64*
```

## Memory/time-efficient Way to Load & Manipulate Data
As everyone has warned me, it is really tricky to handle high-channel count data. I got a taste of this when I worked on the data standardization problem. CJW's approach was to save separate files for each channel and load them in as needed. This is the general philosopy I will take in terms of memory usage; however, I see no need to save them as discrete files.

If you have saved data in an HDF5 format (*e.g.*, v7.3 mat-files), you can access pieces of arrays from files rather than loading the entire file. Note that this is more flexible that Matlab's `matfile()` function.  `matfile()` will only let you access parts of arrays that are at the root of the file. Oftentimes, we organize our data into `structs` and `matfile()` does not allow you to access things nested within structures. In reality, Matlab saves `structs` as part of the hierarchical structure of an HDF5 file. Therefore, you can use HDF5 methods to pull out pieces of parts of structures.

An example will follow below. But my general approach is
1. Load in each 32-bit channel info in one-at-a-time.
2. Bandpass filter the channel data
3. Scale/convert to 16-bit signed integers, and concatenate with other channels from this file in the dataset.
4. *Append* the current file's `Int16` data to the in-progress composite binary file.
5. Rinse and repeat for next file in the dataset.

In this way, the largest array held in memory is the 16-bit (rather than 32-bit) version of the largest original file.

## HDF5 in Matlab
As stated previously, v7.3 mat-files are specifically-formatted HDF5 files. This allows us to use methods in Matlab, Python, Julia, C/C++, *etc.* to read/write pieces of parts of files. In general, you will need to know the size of the binary data you want to load in. This will require a preliminary info-retrieval step.

```matlab
variable_info = h5info(file,'/location/of/var/within/file');
variable_sz = variable_info.Dataspace.Size; % [sz1 sz2]
```

In order to read in a piece of this variable of interest, you can use the following `h5read` syntax.

```matlab
% in general,
variable_piece = h5read(file,'/location/of/var/within/file',[start_dim1 start_dim2],[count_dim1 count_dim2]);

%e.g., single channel (column) of data.
chan = 13;
variable_chan = h5read(file,'/location/of/var/within/file',[1 chan],[variable_sz(1) 1]);
```

This is very fast because the data is being directly addressed with no memory mapping, *etc.*, required. When I have more time, I will wrap user-friendly code around this functionality, specifically for the Stanley Lab data structure I let stagnate for about 9 months.

## Example Preprocessing Script (Matlab)
**You will notice below that KiloSort does a pre-processing step, which includes BP filtering. In the future, I should probably bypass filtering at this stage or disable that part of the KiloSort code.**

Here is my preprocessing script as of this writing (2018/03/28).

```matlab
% name of the binary file for the composite dataset
binFileName = '64Chan_RTXI-180322';

% list the files in order they were collected.
filenames = {'P7_3410_D3_wNoise_1a_post.tdt';'P7_3410D3_wCharac_MB_post.tdt'};
ldDir = 'raw/mat/';
destDir = 'spkPre/sorting/';

scale = 5e5;
nChan = 32;

% BP filter params
ord = 10;
cutoff_lo = 300;
cutoff_hi = 5000;

Vraw_sz = zeros(length(filenames), 2);

for k=1:length(filenames)

    file = [ldDir filenames{k} '.mat'];
    Vraw_info = h5info(file,'/data/streams/Vraw/data/');
    Vraw_sz(k,:) = Vraw_info.Dataspace.Size;

    for chan=1:nChan
        tic()
        Vraw=h5read(file,'/data/streams/Vraw/data',[1 chan],[Vraw_sz(k,1) 1]);

        if k==1 && chan==1
            fs = h5read(file,'/data/streams/Vraw/fs');

            % make a BP filter.
            [b,a] = butter(ord,[cutoff_lo cutoff_hi]/fs*2);
            sys=tf(b,a,'Ts',1/fs);

            %{
            figure;
            h=bodeplot(sys);
            opts=getoptions(h);
            opts.FreqUnits='Hz';
            setoptions(h,opts);
            title('Designed BP Butterworth Filter')
            %}
        end

        % filter data.
        Vfiltd = filtfilt(b,a,double(Vraw));

        % put into appropriate precision.
        if chan==1
        Vdata = int16(Vfiltd * scale)';
        else
        Vdata = [Vdata; int16(Vfiltd * scale)'];
        end

        disp(['Finished chan ' num2str(chan) '/' num2str(nChan) ' in file ' num2str(k) '/' num2str(length(filenames)) ': ' num2str(toc())])
    end %ends channel loop

    tic()
    if k==1
        % create file and write to it.
        fileID = fopen([destDir binFileName '.dat'],'w+');
        fwrite(fileID, Vdata, 'int16');
        fclose(fileID);
    else
        % append to existing data file
        fileID = fopen([destDir binFileName '.dat'],'a+');
        fwrite(fileID, Vdata, 'int16');
        fclose(fileID);
    end
    disp(['Finished writing file ' num2str(k) '/' num2str(length(filenames)) ': ' num2str(toc())])
end %ends file loop

BPfilt_params = struct('filtertype','Butterworth','order',ord,'cutoff_lo',cutoff_lo,'cutoff_hi',cutoff_hi,'B',b,'A',a,'Ts',1/fs);
save([destDir binFileName '.mat'],'-v7.3', 'BPfilt_params', 'ldDir', 'filenames', 'scale', 'fs', 'Vraw_sz');

% e.g., read back in the data
% fileID = fopen([destDir binFileName '.dat'],'r');
% frewind(fileID);
% data = fread(fileID, [nChan sum(Vraw_sz(:,1))],'int16');
```

# Make Channel Map
You need to feed KiloSort a channel map, a logical array that indicates which channels are good or "connected" rather than dead, and the x, y, and shank number coordinates. The x and y coordinates need only be in relative units, not absolute--at least this is what the creator's comments suggest.

1. `chanMap` uint[1 x nChan] : e.g., `chanMap = 1:nChan;`
2. `connected` logical[nChan x 1] : e.g., `connected = true(nChan,1); %if all channels are good`
3. `xcoords` double[1 x nChan]
4. `ycoords` double[1 x nChan]
5. `kcoords` uint[1 x nChan]: e.g., `kcoords = ones(1, nChan); %if all channels on same shank`
6. `fs` double

**You need to save these six variables into a composite `*.mat` (v7.3) file.** You can call it whatever. The name will be configured next.

# Configuration Script
Currently, I do not know what most of these options do because I haven't read the NIPS paper, but you need to create an options `struct` that has the following fields. I will have to get back to you on which (if any) settings we should be changing.

```matlab
clear ops
ops.GPU                 = useGPU; % whether to run this code on an Nvidia GPU (much faster, mexGPUall first)
ops.parfor              = 1; % whether to use parfor to accelerate some parts of the algorithm
ops.verbose             = 1; % whether to print command line progress
ops.showfigures         = 1; % whether to plot figures during optimization

ops.datatype            = 'dat';  % binary ('dat', 'bin') or 'openEphys'
ops.fbinary             = fullfile(fpath, 'sim_binary.dat'); % will be created for 'openEphys'
ops.fproc               = fullfile(fpath, 'temp_wh.dat'); % residual from RAM of preprocessed data
ops.root                = fpath; % 'openEphys' only: where raw files are
% define the channel map as a filename (string) or simply an array
ops.chanMap             = fullfile(fpath, 'chanMap.mat'); % make this file using createChannelMapFile.m
% ops.chanMap = 1:ops.Nchan; % treated as linear probe if unavailable chanMap file

ops.Nfilt               = 64;  % number of clusters to use (2-4 times more than Nchan, should be a multiple of 32)
ops.nNeighPC            = 12; % visualization only (Phy): number of channnels to mask the PCs, leave empty to skip (12)
ops.nNeigh              = 16; % visualization only (Phy): number of neighboring templates to retain projections of (16)

% options for channel whitening
ops.whitening           = 'full'; % type of whitening (default 'full', for 'noSpikes' set options for spike detection below)
ops.nSkipCov            = 1; % compute whitening matrix from every N-th batch (1)
ops.whiteningRange      = 32; % how many channels to whiten together (Inf for whole probe whitening, should be fine if Nchan<=32)

ops.criterionNoiseChannels = 0.2; % fraction of "noise" templates allowed to span all channel groups (see createChannelMapFile for more info).

% other options for controlling the model and optimization
ops.Nrank               = 3;    % matrix rank of spike template model (3)
ops.nfullpasses         = 6;    % number of complete passes through data during optimization (6)
ops.maxFR               = 20000;  % maximum number of spikes to extract per batch (20000)
ops.fshigh              = 200;   % frequency for high pass filtering
% ops.fslow             = 2000;   % frequency for low pass filtering (optional)
ops.ntbuff              = 64;    % samples of symmetrical buffer for whitening and spike detection
ops.scaleproc           = 200;   % int16 scaling of whitened data
ops.NT                  = 128*1024 + ops.ntbuff;% this is the batch size (try decreasing if out of memory)
% for GPU should be multiple of 32 + ntbuff

% the following options can improve/deteriorate results.
% when multiple values are provided for an option, the first two are beginning and ending anneal values,
% the third is the value used in the final pass.
ops.Th               = [4 10 10];    % threshold for detecting spikes on template-filtered data ([6 12 12])
ops.lam              = [5 5 5];   % large means amplitudes are forced around the mean ([10 30 30])
ops.nannealpasses    = 4;            % should be less than nfullpasses (4)
ops.momentum         = 1./[20 400];  % start with high momentum and anneal (1./[20 1000])
ops.shuffle_clusters = 1;            % allow merges and splits during optimization (1)
ops.mergeT           = .1;           % upper threshold for merging (.1)
ops.splitT           = .1;           % lower threshold for splitting (.1)

% options for initializing spikes from data
ops.initialize      = 'fromData';    %'fromData' or 'no'
ops.spkTh           = -6;      % spike threshold in standard deviations (4)
ops.loc_range       = [3  1];  % ranges to detect peaks; plus/minus in time and channel ([3 1])
ops.long_range      = [30  6]; % ranges to detect isolated peaks ([30 6])
ops.maskMaxChannels = 5;       % how many channels to mask up/down ([5])
ops.crit            = .65;     % upper criterion for discarding spike repeates (0.65)
ops.nFiltMax        = 10000;   % maximum "unique" spikes to consider (10000)

% load predefined principal components (visualization only (Phy): used for features)
dd                  = load('PCspikes2.mat'); % you might want to recompute this from your own data
ops.wPCA            = dd.Wi(:,1:7);   % PCs

% options for posthoc merges (under construction)
ops.fracse  = 0.1; % binning step along discriminant axis for posthoc merges (in units of sd)
ops.epu     = Inf;

ops.ForceMaxRAMforDat   = 20e9; % maximum RAM the algorithm will try to use; on Windows it will autodetect.
```

The two things you **definitely** have to change are the following.
1. `opts.fbinary = 'string/with/location/of/binary/file.dat'`
2. `opts.chanMap = 'string/with/location/of/chanMap.mat'`

# Running KiloSort
Before we move forward, add the KiloSort & npy-matlab repositories to your MATLAB path variable.
```matlab
addpath(genpath('path/to/KiloSort')
addpath(genpath('path/to/npy-matlab')
```

Once you have (1) saved your binary data, (2) saved your appropriately formatted channel map, and (3) created your options structure (in that order), actually running KiloSort is very simple.

```matlab
% This part runs the normal Kilosort processing on the simulated data
[rez, DATA, uproj] = preprocessData(ops); % preprocess data and extract spikes for initialization
rez                = fitTemplates(rez, DATA, uproj);  % fit templates iteratively
rez                = fullMPMU(rez, DATA);% extract final spike times (overlapping extraction)

% save python results file for Phy
rezToPhy(rez, '/path/for/sorting/output');
```
This will dump the results of the sorting procedure into `.py` and `.npy` files for reading into `phy`.

# Curating raw sorting results

The dreaded manual sort curation is done using "Cortex Lab"'s [`phy`](https://github.com/kwikteam/phy).

## Installing `miniconda`, `phy` & `phy-contrib`
`phy` is best installed/run in a `conda` environment. I've had problems with these environments in the past, so resisted using `conda`, but I had to suck it up and re-install it. It is certainly the best way to insulate varying versions of python and other packages that are needed to make `phy` work--especially if you use python for other things. At least on macOS you simply [download](https://conda.io/miniconda.html) the install shell script, give it executable permissions and run. Note that you can download the python 3.6, 64-bit version.

```shell
cd path/to/shell/script
chmod +x Miniconda3-latest-MacOSX-x86_64.sh
./Miniconda3-latest-MacOSX-x86_64.sh
```

Now that `conda` is installed, you can set up the `phy` environment. First copy the contents of [this page](https://raw.githubusercontent.com/kwikteam/phy/master/installer/environment.yml) and save it to an environment file. Save it as something like `phy.yml`. Then create the conda environment.
```shell
cd path/to/file
conda env create -f phy.yml
```

Then, you can follow the instructions on the `phy` GitHub page.

```shell
source activate phy  # omit the `source` on Windows
pip install phy phycontrib
```

## Running template-gui
`phy-contrib` has two plugins, one of which is `template-gui`. This is what will be used for the manual curation of KiloSort results.

```shell
cd path/to/output/of/kilosort

source activate phy
phy template-gui params.py
```

Note that running with the above syntax, you must have the binary data file used for sorting in the same directory as the outputs of KiloSort. Otherwise, you will only see the identified template waveforms (rather than real snippets) and you will not be able to use the TraceView, which is one of the most useful features of this GUI.

After this step, I believe everyone else in lab is more expert than I am. [Here is the documentation I am working from](http://phy-contrib.readthedocs.io/en/latest/template-gui/). I will stop here (for now).
[2018/04/04]

