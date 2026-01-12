function varargout = gui(varargin)
% GUI MATLAB code for gui.fig
% GUI, by itself, creates a new GUI or raises the existing singleton.
%
% H = GUI returns the handle to a new GUI or the handle to the existing singleton.
%
% GUI('CALLBACK',hObject,eventData,handles,...) calls the local function named CALLBACK in GUI.M with the given input arguments.
%
% GUI('Property','Value',...) creates a new GUI or raises the existing singleton.
% Starting from the left, property value pairs are applied to the GUI before gui_OpeningFcn gets called.
% An unrecognized property name or invalid value makes property application stop.
% All inputs are passed to gui_OpeningFcn via varargin.
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Last Modified by GUIDE v2.5 16-May-2022 14:07:22

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @gui_OpeningFcn, ...
                   'gui_OutputFcn',  @gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before gui is made visible.
function gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to gui (see VARARGIN)

% Choose default command line output for gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = gui_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
global train
matlabpath = 'D:\\22-3-23';
data = fullfile(matlabpath,'dataset');
train = imageDatastore(data, 'IncludeSubfolders',true,'LabelSource','foldernames');
count = train.countEachLabel;
msgbox('Dataset Loaded Successfully');
% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
global layers
disp('Pre-Trained Model Loaded...');
net = googlenet;
layers = [imageInputLayer([250 540 1])
          net(2:end-3)
          fullyConnectedLayer(2)
          softmaxLayer
          classificationLayer];
msgbox('Pre-Trained Model Loaded Successfully');
% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
global opt training train layers
opt = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 20, ...
    'Verbose', true, 'Plots','training-progress');
training = trainNetwork(train,layers,opt);
msgbox('Trained Completed');
% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
global inp
cd input
[file path] = uigetfile({'*.bmp;*.png;*.jpg'},'Pick an Image File');
if isequal(file,0)
   warndlg('File not selected');
else
   inp = imread(file);
   cd ..
   axes(handles.axes1);
   imshow(inp);
   title('Input Image');
   img=inp;
   if size(inp,3)==1
      Fr=img;
   else
      Fr= im2gray(inp);
   end
   handles.img=img;
end
% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
global inp J
img = im2gray(inp);
J = imgradient(inp);
axes(handles.axes2);
imshow(J);
title('Filtered Image');
% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
img = handles.img;
wavename = 'haar';
[cA,cH,cV,cD] = dwt2(im2double(img),wavename);
[cAA,cAH,cAV,cAD] = dwt2(cA,wavename); % Recompute Wavelet of Approximation Coefs. Level=2
[cAA,cAH;cAV,cAD]; % contacinat
axes(handles.axes4);
imshow([cAA,cAH;cAV,cAD],[],'Colormap',gray);
title('DWT');
% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
global training inp out
out = classify(training,inp);
axes(handles.axes3);
imshow(inp);
title(string(out));
% Update handles structure
guidata(hObject, handles);
