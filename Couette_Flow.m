%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   W7 Version of the Code                         %    by Luis Mora Ortiz 
%   Bachelor Thesis                                %    2021/2022          
%   A Deep Learning Approach to Solve Partial Differential Equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   The Partial Differential Equation are the NAVIER-STOKES' EQUATIONS
%   The Activation function is the ARCTANGENT
%   The Boundary Conditions are DIRICHLET
%   The Regime to Obtain is the COUETTE FLOW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This is the baseline version for the Couette flow regime, which is a
% two-staged optimization by which the first step applies the Boundary
% Condition and then the second step optimizes all of the constraints.
% The data for this problem is contained within the datafile
% "Couette_Flow.mat" which contains one successful solution to the problem

%% Part I.- Starting the Program and General Definitions
% Could be commented due to the large runtime of the algorithm
clear; clc; close all;

%% a.   Describe the dimensions of the problem
% Set the dimensions in the x-direction
xmin =  0;
xmax =  2; % Run x = 2 so that it is a narrow channel
% Set the dimensions in the y-direction
ymin =  0;
ymax =  1;

%% b.   Description of the Boundary Conditions (constraints in space)
% Prescribe the number of boundary conditions at either side (temporal)
xnum = 21; ynum = 21; pnum = 21;
% Set the value of x for the x-Boundary Conditions
x0BC1 = xmin*ones(1,ynum);
x0BC2 = xmax*ones(1,ynum);
% Set the value of y for the x-Boundary Conditions
y0BC1 = linspace(ymin,ymax,ynum);
y0BC2 = linspace(ymax,ymin,ynum);
% Set the value of the function as zero at the BC (Dirichlet BCs)
u0BC1 = linspace(ymin,ymax,ynum);
u0BC2 = linspace(ymax,ymin,ynum);
v0BC1 = zeros(1,ynum);
v0BC2 = zeros(1,ynum);

% Set the value of y for the x-Boundary Conditions
y0BC3 = ymin*ones(1,xnum);
y0BC4 = ymax*ones(1,xnum);
% Set the value of x for the x-Boundary Conditions
x0BC3 = linspace(xmin,xmax,xnum);
x0BC4 = linspace(xmax,xmin,xnum);
% Set the value of the function as zero at the BC (Dirichlet BCs)
u0BC3 = zeros(1,xnum);
u0BC4 =  ones(1,xnum);
v0BC3 = zeros(1,xnum);
v0BC4 = zeros(1,xnum);
% There are no initial conditions, the problem is no longer time dep't
% Let us consider the downstream end to have a  p=0 constraint
y0PC = linspace(ymin,ymax,pnum);
x0PC = xmax*ones(1,pnum);
p0PC = zeros(1,pnum);

%% d.   Generate the final constraints by group ICs and BCs
% Write all the constraints in vector form
% Let us group the Boundary Conditions
XBC = [x0BC1, x0BC3, x0BC2, x0BC4];
YBC = [y0BC1, y0BC3, y0BC2, y0BC4];
UBC = [u0BC1, u0BC3, u0BC2, u0BC4];
VBC = [v0BC1, v0BC3, v0BC2, v0BC4];

%% e.   Generate the full region of internal points by using a mesh grid.
% Set the number of points to have beyond the initial ones.
% Describing the x and y-coordinates:
dx = (xmax-xmin)/(length(x0BC1)-1);
dy = (ymax-ymin)/(length(y0BC1)-1);
% Describing the interior point while including the boundaries in those
% interior points so that they be analyzed as well
Xc = xmin:dx:xmax;      % (xmin+dx):dx:(xmax-dx)
Yc = ymin:dy:ymax;
% Describing the matrices will all internal points XX, YY. (nxn)
XX = zeros(xnum,ynum);
YY = zeros(xnum,ynum);
% Introducing the values in the matrix
for i = 1:length(Xc)
    for j = 1:length(Yc)
        XX(i,j) = Xc(i);
        YY(i,j) = Yc(j);
    end
end
% Describing the time vector tpts and space vector xpts, all in a vector
xpts = zeros(1,xnum*ynum);
ypts = zeros(1,xnum*ynum);
for j = 1:length(Xc)
    for k = 1:length(Yc)
        ii = (k-1)*length(Xc) + j;
        xpts(ii) = XX(j,k);
        ypts(ii) = YY(j,k);
    end
end

%% f. Define the Deep Learning Model
% Let us consider a multilayer perceptron architecture which has L fully
% connected operations and N hidden neurons.
numLayers = 20;
numNeurons = 10;

% The structure will be such that it requires 2 input channels (x and y)
% and delivers a triple output channel u(x,y) v(x,y) p(x,y).
% Let us simplify the components to be passed across the inputs section
% First, the Boundary Conditions
% The variables to include as inputs are the following
% XBC, YBC, UBC, VBC, x0PC, y0PC, p0PC, xpts, and ypts

%% Inserting all of these variables into a structure to be passed around
inputs = struct;
% Boundary Conditions (BC)
inputs.XBC = XBC;
inputs.YBC = YBC;
inputs.UBC = UBC;
inputs.VBC = VBC;
% Pressure Conditions (PC)
inputs.PX = x0PC;
inputs.PY = y0PC;
inputs.PP = p0PC;
% Internal Point Grid
inputs.dataX = xpts';
inputs.dataY = ypts';

% Thus, the parameters of the Neural Network are initialized
parameters = generateParameters(numLayers,numNeurons);
% Redefining the initial parameters in a new structure for it to be kept.
parameters_init = parameters;

%% g.   Select the options for the fmincon algorithm
options = optimoptions('fmincon', ... % Using the multivariate restricted optimization function fmincon
    'HessianApproximation','lbfgs', ... % Using the Lim-Memory BFGS algorithm for the Hessian
    'MaxIterations',1e4, ... % Needs to be sufficiently large so as to ensure convergence
    'MaxFunctionEvaluations',2e4, ... % Needs to be sufficiently large so as to ensure convergence
    'OptimalityTolerance',1e-6, ... % By default considering a tolerance of 0.00001
    'SpecifyObjectiveGradient',true,... % User-defined gradient for the algorithm
    'PlotFcn',@optimplotfirstorderopt,... % Plotting the first-order optimality across iterations
    'Display','iter-detailed'); % Deliver the number of needed iterations

% e. Implement the Model Training for the Hyperbolic Arctangent AF
[parameters_fin,output] = PINN_HE_Dirichlet(inputs,parameters,options);

%% This is the checkpoint for plotting
% Load the document "Report_Couette.mat" if results is what you seek.

%% Plotting only the initial conditions
numPredictions = 1E2+1;
XT = linspace(xmin,xmax,numPredictions);
YT = linspace(ymin,ymax,numPredictions);
XTest = zeros(1,numPredictions^2); 
YTest = zeros(1,numPredictions^2);
for i = 1:width(XT)
    for j = 1:width(YT)
        ii = (i-1)*width(YT)+j;
        XTest(ii) = XT(i);
        YTest(ii) = YT(j);
    end
end

dlXTest = dlarray(XTest,'CB');
dlYTest = dlarray(YTest,'CB');
dlUPred = model(parameters_fin,dlXTest,dlYTest,1);
dlVPred = model(parameters_fin,dlXTest,dlYTest,2);
dlPPred = model(parameters_fin,dlXTest,dlYTest,3);

Xx = [];
Yy = [];
Uu = [];
Vv = [];
Pp = [];
for s = 1:length(dlXTest)
    ii = ceil(s/width(YT)); % iii(s) = ii;
    jj = mod(s,width(YT));  % jjj(s) = jj;
    if ~jj
        jj = width(YT);
    end
    Xx(ii,jj) = XT(ii);
    Yy(ii,jj) = YT(jj);
    Uu(ii,jj) = dlUPred(s);
    Vv(ii,jj) = dlVPred(s);
    Pp(ii,jj) = dlPPred(s);
end
%% Representing the flow
figure
clf
subplot(3,1,1)
surf(Xx,Yy,Uu)
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('u','Interpreter','latex')
title('Horizontal Velocity $U$','Interpreter','latex')
shading interp
colorbar;
subplot(3,1,2)
surf(Xx,Yy,Vv)
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('v','Interpreter','latex')
title('Vertical Velocity $V$','Interpreter','latex')
shading interp
colorbar;
subplot(3,1,3)
surf(Xx,Yy,Pp)
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('u','Interpreter','latex')
title('Pressure $P$','Interpreter','latex')
shading interp
colorbar;

%% Quantifying the error in the outputs.
Uu_exp = Uu*0 + Yy;
Uu_dif = Uu - Uu_exp;
errUu = mse(Uu,Uu_exp)
errVv = mse(Vv,0*Vv)

%% Representing the velocities
k = 30;
figure
clf
subplot(2,2,1)
contourf(Xx,Yy,Uu,k)
hold on 
contour(Xx,Yy,Uu,k)
hold off
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('u','Interpreter','latex')
title('Horizontal Velocity $U$','Interpreter','latex')
shading interp
axis equal
colorbar;
subplot(2,2,2)
contourf(Xx,Yy,Vv,k)
hold on 
contour(Xx,Yy,Vv,k)
hold off
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('v','Interpreter','latex')
title('Vertical Velocity $V$','Interpreter','latex')
shading interp
axis equal
colorbar;
subplot(2,2,3)
contourf(Xx,Yy,Pp,k)
hold on 
contour(Xx,Yy,Pp,k)
hold off
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('u','Interpreter','latex')
title('Pressure $P$','Interpreter','latex')
shading interp
axis equal
colorbar;
subplot(2,2,4)
contourf(Xx,Yy,Uu_dif,k)
hold on 
contour(Xx,Yy,Uu_dif,k)
hold off
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('Err. u','Interpreter','latex')
title('Diff. in $U$','Interpreter','latex')
shading interp
axis equal
colorbar;

%% Extra Functions

%
% fmincon Objective Function
%
function [loss,gradientsV] = objectiveFunction1(parametersU,dlXBC,dlYBC,dlUBC,dlVBC,dlXP,dlYP,dlPP,dlX,dlY,parameterNames,parameterSizes)
    % Rate of Decay
    a = 0.9;
    % Convert parameters to structure of dlarray objects.
    parametersU = dlarray(parametersU);
    parametersu = parameterVectorToStruct(parametersU,parameterNames,parameterSizes);
    % Evaluate model gradients and loss.
    [gradients,loss] = dlfeval(@modelGradients1,parametersu,dlXBC,dlYBC,dlUBC,dlVBC,dlXP,dlYP,dlPP,dlX,dlY);
    % Return loss and gradients for fmincon.
    gradientsV = parameterStructToVector(gradients);
    gradientsV = a*extractdata(gradientsV);
    loss = extractdata(loss);
end
function [loss,gradientsV] = objectiveFunction2(parametersU,dlXBC,dlYBC,dlUBC,dlVBC,dlXP,dlYP,dlPP,dlX,dlY,parameterNames,parameterSizes)
    % Rate of Decay
    a = 0.9;
    % Convert parameters to structure of dlarray objects.
    parametersU = dlarray(parametersU);
    parametersu = parameterVectorToStruct(parametersU,parameterNames,parameterSizes);
    % Evaluate model gradients and loss.
    [gradients,loss] = dlfeval(@modelGradients2,parametersu,dlXBC,dlYBC,dlUBC,dlVBC,dlXP,dlYP,dlPP,dlX,dlY);
    % Return loss and gradients for fmincon.
    gradientsV = parameterStructToVector(gradients);
    gradientsV = a*extractdata(gradientsV);
    loss = extractdata(loss);
end
%
% Generate the Model Gradients Function
%
function [gradients,loss] = modelGradients1(parameters,dlXBC,dlYBC,dlUBC,dlVBC,dlXP,dlYP,dlPP,dlX,dlY)
    % Make predictions with the interior conditions.
    UU = model(parameters,dlX,dlY,1);
    VV = model(parameters,dlX,dlY,2);   
    PP = model(parameters,dlX,dlY,3);
    % Assume a value for the parameter Re
    Re = 1e3;
    % Computing the derivatives with respect to X and Y
    gradientsU = dlgradient(sum(UU,'all'),{dlX,dlY},'EnableHigherDerivatives',true);
    Ux = gradientsU{1};
    Uy = gradientsU{2};
    gradientsV = dlgradient(sum(VV,'all'),{dlX,dlY},'EnableHigherDerivatives',true);
    Vx = gradientsV{1};
    Vy = gradientsV{2};
    gradientsP = dlgradient(sum(PP,'all'),{dlX,dlY},'EnableHigherDerivatives',true);
    Px = gradientsP{1};
    Py = gradientsP{2};
    % Calculate second-order derivatives with respect to X and Y.
    Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true); 
    Uyy = dlgradient(sum(Uy,'all'),dlY,'EnableHigherDerivatives',true); 
    Vxx = dlgradient(sum(Vx,'all'),dlX,'EnableHigherDerivatives',true); 
    Vyy = dlgradient(sum(Vy,'all'),dlY,'EnableHigherDerivatives',true); 
    % Calculate lossF. Enforce the divergence expression.
    f = Ux + Vy;
    % Calculate lossG and lossH. Enforce the Navier-Stokes' equations
    g = UU.*Ux + VV.*Uy + Px - 1/Re*(Uxx + Uyy);
    h = UU.*Vx + VV.*Vy + Py - 1/Re*(Vxx + Vyy);
    zeroTarget = zeros(size(f), 'like', f);
    lossF = mse(f, zeroTarget);
    zeroTarget = zeros(size(g), 'like', g);
    lossG = mse(g, zeroTarget);
    zeroTarget = zeros(size(h), 'like', h);
    lossH = mse(h, zeroTarget);
    % Calculate lossI. Enforce pressure conditions.
    dlPPred = model(parameters,dlXP,dlYP,3);
    lossP = mse(dlPPred, dlPP);
    % Calculate lossB. Enforce boundary conditions. Using Dirichlet BCs
    dlUBCPred = model(parameters,dlXBC,dlYBC,1);
    dlVBCPred = model(parameters,dlXBC,dlYBC,2);
    lossU = mse(dlUBCPred, dlUBC);
    lossV = mse(dlVBCPred, dlVBC);
    % Combine losses.
    loss = lossP + lossU + lossV;
    % loss = lossF + lossG + lossH + lossP + lossU + lossV;
    % Calculate gradients with respect to the learnable parameters.
    gradients = dlgradient(loss,parameters);
end
function [gradients,loss] = modelGradients2(parameters,dlXBC,dlYBC,dlUBC,dlVBC,dlXP,dlYP,dlPP,dlX,dlY)
    % Make predictions with the interior conditions.
    UU = model(parameters,dlX,dlY,1);
    VV = model(parameters,dlX,dlY,2);   
    PP = model(parameters,dlX,dlY,3);
    % Assume a value for the parameter Re
    Re = 1e2;
    % Computing the derivatives with respect to X and Y
    gradientsU = dlgradient(sum(UU,'all'),{dlX,dlY},'EnableHigherDerivatives',true);
    Ux = gradientsU{1};
    Uy = gradientsU{2};
    gradientsV = dlgradient(sum(VV,'all'),{dlX,dlY},'EnableHigherDerivatives',true);
    Vx = gradientsV{1};
    Vy = gradientsV{2};
    gradientsP = dlgradient(sum(PP,'all'),{dlX,dlY},'EnableHigherDerivatives',true);
    Px = gradientsV{1};
    Py = gradientsV{2};
    % Calculate second-order derivatives with respect to X and Y.
    Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true); 
    Uyy = dlgradient(sum(Uy,'all'),dlY,'EnableHigherDerivatives',true); 
    Vxx = dlgradient(sum(Vx,'all'),dlX,'EnableHigherDerivatives',true); 
    Vyy = dlgradient(sum(Vy,'all'),dlY,'EnableHigherDerivatives',true); 
    % Calculate lossF. Enforce the divergence expression.
    f = Ux + Vy;
    % Calculate lossG and lossH. Enforce the Navier-Stokes' equations
    g = UU.*Ux + VV.*Uy + Px - 1/Re*(Uxx + Uyy);
    h = UU.*Vx + VV.*Vy + Py - 1/Re*(Vxx + Vyy);
    zeroTarget = zeros(size(f), 'like', f);
    lossF = mse(f, zeroTarget);
    zeroTarget = zeros(size(g), 'like', g);
    lossG = mse(g, zeroTarget);
    zeroTarget = zeros(size(h), 'like', h);
    lossH = mse(h, zeroTarget);
    % Calculate lossI. Enforce pressure conditions.
    dlPPred = model(parameters,dlXP,dlYP,3);
    lossP = mse(dlPPred, dlPP);
    % Calculate lossB. Enforce boundary conditions. Using Dirichlet BCs
    dlUBCPred = model(parameters,dlXBC,dlYBC,1);
    dlVBCPred = model(parameters,dlXBC,dlYBC,2);
    lossU = mse(dlUBCPred, dlUBC);
    lossV = mse(dlVBCPred, dlVBC);
    % Combine losses.
    % loss = lossG + lossP + lossU + lossV;
    loss = lossF + lossG + lossH + lossP + lossU + lossV;
    % Calculate gradients with respect to the learnable parameters.
    gradients = dlgradient(loss,parameters);
end
%
% Generate the Model for the Full Network
%
function dlU = model(parameters,dlX,dlY,a)
    dlXT = [dlX;dlY];
    numLayers = numel(fieldnames(parameters))/6;
    % First fully connect operation.
    weights = parameters.("fc1_"+ a + "_Weights");
    bias = parameters.("fc1_"+ a + "_Bias");
    dlU = fullyconnect(dlXT,weights,bias);
    % tanh and fully connect operations for remaining layers.
    for i=2:numLayers
        name = "fc" + i + "_" + a;
        dlU = tanh(dlU);
        weights = parameters.(name + "_Weights");
        bias = parameters.(name + "_Bias");
        dlU = fullyconnect(dlU, weights, bias);
    end
end
%
% Initialize He and Initialize Zero for the Parameters
%
function parameter = initializeHe(sz,numIn,className)
    arguments
        sz
        numIn
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
    sz = [numNeurons 2];
    parameters.fc1_1_Weights = initializeHe(sz,2);
    parameters.fc1_1_Bias = initializeZeros([numNeurons 1]);
    parameters.fc1_2_Weights = initializeHe(sz,2);
    parameters.fc1_2_Bias = initializeZeros([numNeurons 1]);
    parameters.fc1_3_Weights = initializeHe(sz,2);
    parameters.fc1_3_Bias = initializeZeros([numNeurons 1]);
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
        parameters.(name + "_" + 1 + "_Weights") = initializeHe(sz,numIn);
        parameters.(name + "_" + 1 + "_Bias") = initializeZeros([numNeurons 1]);
        parameters.(name + "_" + 2 + "_Weights") = initializeHe(sz,numIn);
        parameters.(name + "_" + 2 + "_Bias") = initializeZeros([numNeurons 1]);
        parameters.(name + "_" + 3 + "_Weights") = initializeHe(sz,numIn);
        parameters.(name + "_" + 3 + "_Bias") = initializeZeros([numNeurons 1]);
    end
% c.   Generating the parameters for the last fully connect operation.
    sz = [1 numNeurons];
    % The inputs for the last connect operation are also the number of neurons
    numIn = numNeurons; 
    % Initializing the parameters for the connect operation
    parameters.("fc" + numLayers + "_" + 1 + "_Weights") = initializeHe(sz,numIn);
    parameters.("fc" + numLayers + "_" + 1 + "_Bias") = initializeZeros([1 1]);
    parameters.("fc" + numLayers + "_" + 2 + "_Weights") = initializeHe(sz,numIn);
    parameters.("fc" + numLayers + "_" + 2 + "_Bias") = initializeZeros([1 1]);
    parameters.("fc" + numLayers + "_" + 3 + "_Weights") = initializeHe(sz,numIn);
    parameters.("fc" + numLayers + "_" + 3 + "_Bias") = initializeZeros([1 1]);
end

function [parameters,output] = PINN_HE_Dirichlet(in,parameters,options)
% Part III.- Train the Network Using fmincon
% Generate the vector with the parameters to be inserted in the algorithm.
[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters); 
parametersV = double(extractdata(parametersV));
% Boundary Conditions (BC)
dlXBC = dlarray(in.XBC,'CB');
dlYBC = dlarray(in.YBC,'CB');
dlUBC = dlarray(in.UBC,'CB');
dlVBC = dlarray(in.VBC,'CB');
% Pressure Conditions (PC)
dlXP = dlarray(in.PX,'CB');
dlYP = dlarray(in.PY,'CB');
dlPP = dlarray(in.PP,'CB');
% Internal Point Grid
dlX = dlarray(in.dataX','CB');
dlY = dlarray(in.dataY','CB');

% Define the objective function through the data obtained beforehand
objFun = @(parameters) objectiveFunction1(parameters,dlXBC,dlYBC,dlUBC,dlVBC,...
    dlXP,dlYP,dlPP,dlX,dlY,parameterNames,parameterSizes);

% Run the fmincon algorithm and obtain the total runtime of the algorithm
tic
%[x,fval,exitflag,output,lambda,grad,hessian]
[parametersV,~,~,output,~,~,~] = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
toc

% Define the objective function through the data obtained beforehand
objFun = @(parameters) objectiveFunction2(parameters,dlXBC,dlYBC,dlUBC,dlVBC,...
    dlXP,dlYP,dlPP,dlX,dlY,parameterNames,parameterSizes);

% Run the fmincon algorithm and obtain the total runtime of the algorithm
tic
%[x,fval,exitflag,output,lambda,grad,hessian]
[parametersV,~,~,output,~,~,~] = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
toc
% Revert the optimized version of the parameters to a struct. 
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
end