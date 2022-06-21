%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   W7 Version of the Code                         %    by Luis Mora Ortiz 
%   Bachelor Thesis                                %    2021/2022          
%   A Deep Learning Approach to Solve Partial Differential Equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   The Partial Differential Equation is the HEAT EQUATION
%   The Activation function is the ARCTANGENT
%   The Boundary Conditions are DIRICHLET
%   The Dimension of the problem is TWO-DIMENSIONAL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This is the basic implementation of the solver for the wave equation in
% 2D with Dirichlet Boundary Conditions used as the baseline to develop
% more complex codes.

%% Part I.- Starting the Program and General Definitions
% Could be commented due to the large runtime of the algorithm
clear; clc; close all;

%% a.   Describe the dimensions of the problem
% Set the dimensions in the x-direction
xmin = -1;
xmax =  1;
% Set the dimensions in the y-direction
ymin = -1;
ymax =  1;
% Set the maximum time dimension
tmax =  1;

%% b.   Description of the Boundary Conditions (constraints in space)
% Prescribe the number of boundary conditions at either side (temporal)
tnum = 11; xnum = 31;
% Set the value of x for the x-Boundary Conditions
x0BC1x = xmin*ones(tnum,xnum);
x0BC2x = xmax*ones(tnum,xnum);
% Set the value of y for the x-Boundary Conditions
y0BC1xi = linspace(ymin,ymax,tnum);
y0BC2xi = linspace(ymin,ymax,tnum);
% Set the value of t for the Boundary Conditions (provided a maximum time)
t0BC1xi = linspace(0,tmax,xnum);
t0BC2xi = linspace(0,tmax,xnum);
t0BC1x = []; t0BC2x = []; y0BC1x = []; y0BC2x = [];
for i = 1:tnum
    t0BC1x = [t0BC1x; t0BC1xi];
    t0BC2x = [t0BC2x; t0BC2xi];
end
for i = 1:xnum
    y0BC1x = [y0BC1x, y0BC1xi'];
    y0BC2x = [y0BC2x, y0BC2xi'];
end
% Set the value of the function as zero at the BC (Dirichlet BCs)
u0BC1x = zeros(tnum,xnum);
u0BC2x = zeros(tnum,xnum);

% Set the value of y for the x-Boundary Conditions
y0BC1y = ymin*ones(tnum,xnum);
y0BC2y = ymax*ones(tnum,xnum);
% Set the value of x for the x-Boundary Conditions
x0BC1yi = linspace(xmin,xmax,tnum);
x0BC2yi = linspace(xmin,xmax,tnum);
% Set the value of t for the Boundary Conditions (provided a maximum time)
t0BC1yi = linspace(0,tmax,xnum);
t0BC2yi = linspace(0,tmax,xnum);
t0BC1y = []; t0BC2y = []; x0BC1y = []; x0BC2y = [];
for i = 1:tnum
    t0BC1y = [t0BC1y; t0BC1yi];
    t0BC2y = [t0BC2y; t0BC2yi];
end
for i = 1:xnum
    x0BC1y = [x0BC1y, x0BC1yi'];
    x0BC2y = [x0BC2y, x0BC2yi'];
end
% Set the value of the function as zero at the BC (Dirichlet BCs)
u0BC1y = zeros(tnum,xnum);
u0BC2y = zeros(tnum,xnum);

%% c.   Description of the Initial Conditions (constraints in time)
% Prescribe the number of initial condition points (spatial)
xnum = 31; ynum = 31;
tnum  = 11;
% Set the value of x for the Initial Conditions (linearized to [-1,1])
x0ICi = linspace(xmin,xmax,xnum);
% Set the value of y for the Initial Conditions (linearized to [-1,1])
y0ICi = linspace(ymin,ymax,ynum);
x0IC = []; y0IC = [];
for i = 1:ynum
    x0IC = [x0IC; y0ICi];
end
for i = 1:xnum
    y0IC = [y0IC, y0ICi'];
end
% Set the value of t for the Initial Conditions
t0IC = zeros(xnum,ynum);
% Set the value of the Initial Condition as a bell curve for this scenario
alpha = 5^2;
u0IC = exp(-alpha*(x0IC.^2+y0IC.^2));

%% d.   Generate the final constraints by group ICs and BCs
% Write all the constraints in vector form
% Let us group the x-coordinate BCs
XBCx = [x0BC1x x0BC2x];
YBCx = [y0BC1x y0BC2x];
TBCx = [t0BC1x t0BC2x];
UBCx = [u0BC1x u0BC2x];
% Let us group the y-coordinate BCs
XBCy = [x0BC1y x0BC2y];
YBCy = [y0BC1y y0BC2y];
TBCy = [t0BC1y t0BC2y];
UBCy = [u0BC1y u0BC2y];
% Let us group the Initial Conditions
XICi = x0IC;
YICi = y0IC;
TICi = t0IC;
UICi = u0IC;

%% e.   Generate the full region of internal points by using a mesh grid.
% Set the number of points to have beyond the initial ones.
% Describing the x and t-coordinates:
dt = (tmax-0)/(length(t0BC1x)-1);
dx = (xmax-xmin)/(length(x0IC)-1);
dy = (ymax-ymin)/(length(y0IC)-1);

Xc = xmin:dx:xmax;      % (xmin+dx):dx:(xmax-dx)
Yc = ymin:dy:ymax;
Tc = 0:dt:tmax;         % (0   +dt):dt:(tmax-dt)
% Describing the matrices will all internal points XX, YY and TT. (nxnxn)
TT = zeros(tnum,xnum,ynum);
XX = zeros(tnum,xnum,ynum);
YY = zeros(tnum,xnum,ynum);

for i = 1:length(Tc)
    for j = 1:length(Xc)
        for k = 1:length(Yc)
            TT(i,j,k) = Tc(i);
            XX(i,j,k) = Xc(j);
            YY(i,j,k) = Yc(k);
        end
    end
end

% Describing the time vector tpts and space vector xpts, all in a vector
tpts = zeros(1,xnum*ynum*tnum);
xpts = zeros(1,xnum*ynum*tnum);
ypts = zeros(1,xnum*ynum*tnum);
for i = 1:length(Tc)
    for j = 1:length(Xc)
        for k = 1:length(Yc)
            ii = (k-1)*length(Tc)*length(Xc) + (j-1)*length(Tc) + i;
            tpts(ii) = TT(i,j,k);
            xpts(ii) = XX(i,j,k);
            ypts(ii) = YY(i,j,k);
        end
    end
end

%% f. Define the Deep Learning Model
% Let us consider a multilayer perceptron architecture which has L fully
% connected operations and N hidden neurons.
numLayers = 10;
numNeurons = 20;

% The structure will be such that it requires 2 input channels (x and t)
% and delivers a single output channel u(x,t).

% Let us simplify the components to be passed across the inputs section
% First, the Boundary Conditions
XBCi = [XBCx; XBCy];
YBCi = [YBCx; YBCy];
TBCi = [TBCx; TBCy];
UBCi = [UBCx; UBCy];
XBC = zeros(1,height(XBCi)*width(XBCi));
YBC = zeros(1,height(XBCi)*width(XBCi)); 
TBC = zeros(1,height(XBCi)*width(XBCi));
UBC = zeros(1,height(XBCi)*width(XBCi));
for i = 1:height(XBCi)
    for j = 1:width(XBCi)
        ii = (i-1)*width(XBCi)+j;
        XBC(ii) = XBCi(i,j);
        YBC(ii) = YBCi(i,j);
        TBC(ii) = TBCi(i,j);
        UBC(ii) = UBCi(i,j);
    end
end
% Then, the Initial Conditions
XIC = zeros(1,height(XICi)*width(XICi));
YIC = zeros(1,height(XICi)*width(XICi)); 
TIC = zeros(1,height(XICi)*width(XICi));
UIC = zeros(1,height(XICi)*width(XICi));
for i = 1:height(XICi)
    for j = 1:width(XICi)
        ii = (i-1)*width(XICi)+j;
        XIC(ii) = XICi(i,j);
        YIC(ii) = YICi(i,j);
        TIC(ii) = TICi(i,j);
        UIC(ii) = UICi(i,j);
    end
end

%% g. Plotting the Boundary Conditions, Initial Conditions, and Inside Pts

figure
plot3(xpts,ypts,tpts,'g*')
hold on
plot3(XIC,YIC,TIC,'b*')
plot3(XBC,YBC,TBC,'r*')
hold off
xlabel('x-coordinate','Interpreter','latex')
ylabel('y-coordinate','Interpreter','latex')
zlabel('t-coordinate','Interpreter','latex')
grid minor
legend('Collocation Points','Initial Condition','Boundary Conditions',...
    'Location','best','Interpreter','latex')


%% Inserting all of these variables into a structure to be passed around
inputs = struct;
% Boundary Conditions (BC)
inputs.XBC = XBC;
inputs.YBC = YBC;
inputs.TBC = TBC;
inputs.UBC = UBC;
% Initial Conditions (IC)
inputs.XIC = XIC;
inputs.YIC = YIC;
inputs.TIC = TIC;
inputs.UIC = UIC;
% Internal Point Grid
inputs.dataX = xpts';
inputs.dataY = ypts';
inputs.dataT = tpts';

% Thus, the parameters of the Neural Network are initialized
parameters = generateParameters(numLayers,numNeurons);
% Redefining the initial parameters in a new structure for it to be kept.
parameters_init = parameters;

% g.   Select the options for the fmincon algorithm
options = optimoptions('fmincon', ... % Using the multivariate restricted optimization function fmincon
    'HessianApproximation','lbfgs', ... % Using the Lim-Memory BFGS algorithm for the Hessian
    'MaxIterations',200, ... % Needs to be sufficiently large so as to ensure convergence
    'MaxFunctionEvaluations',200, ... % Needs to be sufficiently large so as to ensure convergence
    'OptimalityTolerance',1e-3, ... % By default considering a tolerance of 0.0001
    'SpecifyObjectiveGradient',true,... % User-defined gradient for the algorithm
    'Display','iter-detailed'); % Deliver the number of needed iterations

% e. Implement the Model Training for the Hyperbolic Arctangent AF
[parameters_fin,output] = PINN_HE_Dirichlet(inputs,parameters,options,1);

%% Let us plot some results:
% Let us first determine a set of particular time instances at which to
% study the model.
tTest = [0.00 0.10 0.25 0.50 0.75 1.00];
numPredictions = 1E2+1;
XT = linspace(-1,1,numPredictions);
YT = linspace(-1,1,numPredictions);
XTest = zeros(1,numPredictions^2); 
YTest = zeros(1,numPredictions^2);
for i = 1:width(XT)
    for j = 1:width(YT)
        ii = (i-1)*width(YT)+j;
        XTest(ii) = XT(i);
        YTest(ii) = YT(j);
    end
end

figure
for i = 1:numel(tTest)
    t = tTest(i);
    TTest = t*ones(size(XTest));
    % Make predictions.
    dlXTest = dlarray(XTest,'CB');
    dlYTest = dlarray(YTest,'CB');
    dlTTest = dlarray(TTest,'CB');
    dlUPred = model(parameters_fin,dlXTest,dlYTest,dlTTest,1);
    Xx = [];
    Yy = [];
    Uu = [];
    for s = 1:length(dlXTest)
        ii = ceil(s/width(YT)); % iii(s) = ii;
        jj = mod(s,width(YT));  % jjj(s) = jj;
        if ~jj
            jj = width(YT);
        end
        Xx(ii,jj) = XT(ii);
        Yy(ii,jj) = YT(jj);
        Uu(ii,jj) = dlUPred(s);
    end

    % Plot predictions.
    subplot(3,2,i)
    surf(Yy,Xx,Uu);
    title("t = " + t,'Interpreter','latex');
    colorbar
end

%%
for i = 0:50
    for j = 0:50
        aa(i+1,j+1) = integral2(@(x,y) exp(-alpha*(x.^2+y.^2)).*cos(i*pi*x/2).*cos(j*pi*y/2),0,1,0,1);
    end
end

figure
for i = 1:numel(tTest)
    t = tTest(i);
    TTest = t*ones(size(XTest));
    alpha = 25;     k = 0.01;
    UAnal = solveWaveEquation(XTest,YTest,TTest,alpha,k,aa);
    Uu = [];
    for s = 1:length(dlXTest)
        ii = ceil(s/width(YT)); % iii(s) = ii;
        jj = mod(s,width(YT));  % jjj(s) = jj;
        if ~jj
            jj = width(YT);
        end
        Uu(ii,jj) = UAnal(s);
    end

    % Plot predictions.
    subplot(3,2,i)
    surf(Xx,Yy,Uu);
    title("t = " + t,'Interpreter','latex');
    colorbar
end

%% Post comparison of PINN - expected and error (div by mse)
figure
t = [0, 0.25, 1];
for i = 1:3
    TTest = t(i)*ones(size(XTest));
    % Make predictions.
    dlXTest = dlarray(XTest,'CB');
    dlYTest = dlarray(YTest,'CB');
    dlTTest = dlarray(TTest,'CB');
    dlUPred = model(parameters_fin,dlXTest,dlYTest,dlTTest,1);
    Xx = [];
    Yy = [];
    UP = [];
    for s = 1:length(dlXTest)
        ii = ceil(s/width(YT)); % iii(s) = ii;
        jj = mod(s,width(YT));  % jjj(s) = jj;
        if ~jj
            jj = width(YT);
        end
        Xx(ii,jj) = XT(ii);
        Yy(ii,jj) = YT(jj);
        UP(ii,jj) = dlUPred(s);
    end
    % Obtaining the analytical expression
    TTest = t(i)*ones(size(XTest));
    alpha = 25;     k = 0.01;
    UAnal = solveWaveEquation(XTest,YTest,TTest,alpha,1,aa);
    UA = [];
    for s = 1:length(dlXTest)
        ii = ceil(s/width(YT)); % iii(s) = ii;
        jj = mod(s,width(YT));  % jjj(s) = jj;
        if ~jj
            jj = width(YT);
        end
        UA(ii,jj) = UAnal(s);
    end

    % Plot predictions.
    subplot(3,3,3*i-2)
    surf(Xx,Yy,UP);
    title("PINN Model at t = " + t(i),'Interpreter','latex');
    colorbar
    caxis([min(UAnal),max(UAnal)]);
    shading interp

    
    % Plot predictions.
    subplot(3,3,3*i-1)
    surf(Xx,Yy,UA);
    title("Analytical Solution at t = " + t(i),'Interpreter','latex');
    colorbar
    caxis([min(UAnal),max(UAnal)]);
    shading interp

    max(max(UA)')
    subplot(3,3,3*i)
    Uv = abs(UA-UP)./max(max(UA)');
    surf(Xx,Yy,Uv);
    title("$L_{\infty}$ norm error at t = " + t(i),'Interpreter','latex');
    colorbar
    shading interp
end

save WE_unity

%% Extra Functions
%
% Heat Equation Solver
%
function u = solveWaveEquation(x,y,t,alpha,k,aa)
arguments
    x
    y
    t
    alpha = 25;
    k = 0.01;
    aa = 0;
end 
c = sqrt(k);
% The functional form of the solution is dependent on these two values:
% \sum_i ai*sin(i pi x)*exp(-k*(i*pi)^2)
% Let us obtain each of the ai by Fourier transform
% \int_{-1}^1 exp(-alpha*x^2)*sin(i*pi*x) dx = ai * pi/2
    if aa == 0
        for i = 0:10
            for j = 0:10
                aa(i+1,j+1) = integral2(@(x,y) exp(-alpha*(x.^2+y.^2)).*cos(i*pi*x/2).*cos(j*pi*y/2),0,1,0,1);
            end
        end
    end
    u = 0;
    for i = 0:(height(aa)-1)
        for j = 0:(width(aa)-1)
            A = (2 - (i==0)).*(2 - (j==0));
            u = u + A/4*aa(i+1,j+1).*cos(c.*sqrt((i*pi/2)^2+(j*pi/2)^2).*t).*cos(i*pi.*x/2).*cos(j*pi.*y/2);
        end
    end
% u = 0.5*exp(-alpha*((x-c*t).^2+(y-c*t).^2)) + 0.5*exp(-alpha*((x+c*t).^2+(y+c*t).^2));
end
function u = solveHeatEquation(x,y,t,alpha,k,aa)
arguments
    x
    y
    t
    alpha = 25;
    k = 0.01;
    aa = 0
end
% The functional form of the solution is dependent on these two values:
% \sum_i ai*sin(i pi x)*exp(-k*(i*pi)^2)
% Let us obtain each of the ai by Fourier transform
% \int_{-1}^1 exp(-alpha*x^2)*sin(i*pi*x) dx = ai * pi/2
    if aa == 0
        for i = 0:50
            for j = 0:50
                A = (2 - (i==0)).*(2 - (j==0));
                aa(i+1,j+1) = integral2(@(x,y) exp(-alpha*(x.^2+y.^2)).*cos(i*pi*x/2).*cos(j*pi*y/2),0,1,0,1);
            end
        end
    end
    u = 0;
    for i = 0:(height(aa)-1)
        for j = 0:(width(aa)-1)
            A = (2 - (i==0)).*(2 - (j==0));
            u = u + A/4*aa(i+1,j+1).*exp(-k*(pi/2)^2.*(i^2+j^2).*t).*cos(i*pi.*x/2).*cos(j*pi.*y/2);
        end
    end
end
%
% fmincon Objective Function
%
function [loss,gradientsV] = objectiveFunction(parametersV,dlX,dlY,dlT,dlXIC,dlYIC,dlTIC,dlUIC,dlXBC,dlYBC,dlTBC,dlUBC,parameterNames,parameterSizes,af)
    % Rate of Decay
    a = 0.9;
    % Convert parameters to structure of dlarray objects.
    parametersV = dlarray(parametersV);
    parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
    % Evaluate model gradients and loss.
    [gradients,loss] = dlfeval(@modelGradients,parameters,dlX,dlY,dlT,dlXIC,dlYIC,dlTIC,dlUIC,dlXBC,dlYBC,dlTBC,dlUBC,af);
    % Return loss and gradients for fmincon.
    gradientsV = parameterStructToVector(gradients);
    gradientsV = a*extractdata(gradientsV);
    loss = extractdata(loss);
end
%
% Generate the Model Gradients Function
%
function [gradients,loss] = modelGradients(parameters,dlX,dlY,dlT,dlXIC,dlYIC,dlTIC,dlUIC,dlXBC,dlYBC,dlTBC,dlUBC,a)
    % Make predictions with the initial conditions.
    UU = model(parameters,dlX,dlY,dlT,a);
    % Assume a value for the parameter k
    k = 0.01;
    % Calculate derivatives with respect to X and T.
    gradientsU = dlgradient(sum(UU,'all'),{dlX,dlY,dlT},'EnableHigherDerivatives',true); % +dlXBC+dlXIC//+dlTIC+dlTBC
    Ux = gradientsU{1};
    Uy = gradientsU{2};
    Ut = gradientsU{3};
    % Calculate second-order derivatives with respect to X.
    Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true); % +dlXBC+dlXIC
    Uyy = dlgradient(sum(Uy,'all'),dlY,'EnableHigherDerivatives',true); % +dlXBC+dlXIC
     % Calculate second-order derivatives with respect to T.
    Utt = dlgradient(sum(Ut,'all'),dlT,'EnableHigherDerivatives',true); % +dlXBC+dlXIC
    % Calculate lossF. Enforce Burger's equation.
    f = Utt - k.*(Uxx + Uyy);
    zeroTarget = zeros(size(f), 'like', f);
    lossF = mse(f, zeroTarget);
    % Calculate lossI. Enforce initial conditions.
    dlUICPred = model(parameters,dlXIC,dlYIC,dlTIC,a);
    Un = dlgradient(sum(dlUICPred,'all'),dlTIC,'EnableHigherDerivatives',true);
    lossI = mse(dlUICPred, dlUIC);
    lossJ = mse(Un,0*dlUIC);
    % Calculate lossB. Enforce boundary conditions. Using Dirichlet BCs
    dlUBCPred = model(parameters,dlXBC,dlYBC,dlTBC,a);
    % Un = dlgradient(sum(dlUBCPred,'all'),dlXBC,'EnableHigherDerivatives',true);
    lossB = mse(dlUBCPred, dlUBC);
    % Combine losses.
    loss = lossF + lossI + lossJ + lossB;
    % Calculate gradients with respect to the learnable parameters.
    gradients = dlgradient(loss,parameters);
end
%
% Generate the Model for the Full Network
%
function dlU = model(parameters,dlX,dlY,dlT,a)
    arguments
        parameters
        dlX
        dlY
        dlT
        a = 1;
    end
    dlXT = [dlX;dlY;dlT];
    numLayers = numel(fieldnames(parameters))/2;
    % First fully connect operation.
    weights = parameters.fc1_Weights;
    bias = parameters.fc1_Bias;
    dlU = fullyconnect(dlXT,weights,bias);
    % tanh and fully connect operations for remaining layers.
    for i=2:numLayers
        name = "fc" + i;
        switch a
            case 1 % Hyperbolic Arctangent
                dlU = tanh(dlU);
            case 2 % Rectified Linear Unit
                dlU = max(0,dlU);
%             case 2 % Exponential Rectified Linear Unit
%                 if dlU <= 0
%                     dlU = 0.01*(exp(dlU)-1);
%                 end
%             case 2
%                 dlU = sqrt(1 + ep.^2*dlU.^2);
            case 3 % Sigmoid Activation Function
                dlU = 1./(1+exp(-dlU));
%             case 3
%                 dlU = exp(-ep.^2*dlU.^2);
        end
        weights = parameters.(name + "_Weights");
        bias = parameters.(name + "_Bias");
        dlU = fullyconnect(dlU, weights, bias);
    end
end
%
% Initialize He and Initialize Zero for the Parameters
%
function parameter = initializeHe(sz,numIn,i,className)
    arguments
        sz
        numIn
        i = 0
        className = 'single'
    end
    parameter = sqrt(2/numIn) * randn(sz,className);
    % This algorithm does not converge unless the initial definition has
    % some random arbitrary values, and thus, it might be more sensible to
    % set a common seed so that the convergence is relative to that choice
    parameter = dlarray(parameter);
end

function parameter = initializeZeros(sz,className)
    arguments
        sz
        className = 'single'
    end
    parameter = zeros(sz,className);
    parameter = dlarray(parameter);
end
%
% Parameter Struct to Vector and Vector to Struct
%
function parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes)
% parameterVectorToStruct converts a vector of parameters with specified
% names and sizes to a struct.
    parameters = struct;
    numFields = numel(parameterNames);
    count = 0;
    for i = 1:numFields
        numElements = prod(parameterSizes{i});
        parameter = parametersV(count+1:count+numElements);
        parameter = reshape(parameter,parameterSizes{i});
        parameters.(parameterNames{i}) = parameter;
        count = count + numElements;
    end
end

function [parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters)
% parameterStructToVector converts a struct of learnable parameters to a
% vector and also returns the parameter names and sizes.
    % Parameter names.
    parameterNames = fieldnames(parameters);
    % Determine parameter sizes.
    numFields = numel(parameterNames);
    parameterSizes = cell(1,numFields);
    for i = 1:numFields
        parameter = parameters.(parameterNames{i});
        parameterSizes{i} = size(parameter);
    end
    % Calculate number of elements per parameter.
    numParameterElements = cellfun(@prod,parameterSizes);
    numParamsTotal = sum(numParameterElements);
    % Construct vector
    parametersV = zeros(numParamsTotal,1,'like',parameters.(parameterNames{1}));
    count = 0;
    for i = 1:numFields
        parameter = parameters.(parameterNames{i});
        numElements = numParameterElements(i);
        parametersV(count+1:count+numElements) = parameter(:);
        count = count + numElements;
    end
end
%
% Function to generate the parameters struct of the function
%
function parameters = generateParameters(numLayers,numNeurons)
% Let us generate a structure for the parameters of the operation so that
% we might describe a variable that depends on the manually inputted
% neurons and layers.
    parameters = struct;
% a.   Generating the parameters for the first operation.
    % The inputs are the initial values of x and t (2-dimensions) for each 
    % of the number of total neurons, therefore numNeurons
    sz = [numNeurons 3];
    parameters.fc1_Weights = initializeHe(sz,3);
    parameters.fc1_Bias = initializeZeros([numNeurons 1]);
% b.   Generating the parameters for the intermediate connect operations.
    % The inputs are the numNeurons and the outputs are also numNeurons
    for layerNumber=2:numLayers-1
        name = "fc"+layerNumber;
        % It will be a nxn matrix in which element i j refers to the weight of
        % element i of the previous layer to the element j of the new one
        sz = [numNeurons numNeurons];
        % The number of inputs is the number of neurons in the previous layer
        numIn = numNeurons;
        % Initializing the parameters for the connect operation
        parameters.(name + "_Weights") = initializeHe(sz,numIn);
        parameters.(name + "_Bias") = initializeZeros([numNeurons 1]);
    end
% c.   Generating the parameters for the last fully connect operation.
    sz = [1 numNeurons];
    % The inputs for the last connect operation are also the number of neurons
    numIn = numNeurons; 
    % Initializing the parameters for the connect operation
    parameters.("fc" + numLayers + "_Weights") = initializeHe(sz,numIn);
    parameters.("fc" + numLayers + "_Bias") = initializeZeros([1 1]);
end

function [parameters,output] = PINN_HE_Dirichlet(in,parameters,options,af)
% Part III.- Train the Network Using fmincon
% Generate the vector with the parameters to be inserted in the algorithm.
[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters); 
parametersV = double(extractdata(parametersV));

% Generate the arrays with the data for loss function minimization

% Boundary Conditions (BC)
dlXBC = dlarray(in.XBC,'CB');
dlYBC = dlarray(in.YBC,'CB');
dlTBC = dlarray(in.TBC,'CB');
dlUBC = dlarray(in.UBC,'CB');
% Initial Conditions (IC)
dlXIC = dlarray(in.XIC,'CB');
dlYIC = dlarray(in.YIC,'CB');
dlTIC = dlarray(in.TIC,'CB');
dlUIC = dlarray(in.UIC,'CB');
% Internal Point Grid
dlX = dlarray(in.dataX','CB');
dlY = dlarray(in.dataY','CB');
dlT = dlarray(in.dataT','CB');

% Define the objective function through the data obtained beforehand
objFun = @(parameters) objectiveFunction(parameters,dlX,dlY,dlT,...
    dlXIC,dlYIC,dlTIC,dlUIC,dlXBC,dlYBC,dlTBC,dlUBC,parameterNames,parameterSizes,af);

% Run the fmincon algorithm and obtain the total runtime of the algorithm
tic
%[x,fval,exitflag,output,lambda,grad,hessian]
[parametersV,~,~,output,~,~,~] = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
toc
% Revert the optimized version of the parameters to a struct. 
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
end
