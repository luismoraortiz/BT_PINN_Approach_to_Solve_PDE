%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Final Version of the OptimComps Code           %    by Luis Mora Ortiz 
%   Bachelor Thesis                                %    2021/2022          
%   A Deep Learning Approach to Solve Partial Differential Equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Considerations to be made:
%   The PDE of the problem is the Heat equation
%   The BCs are Dirichlet Boundary Conditions
%   This Code will compare Optimization methods to solve the problem.

% This method consists on the comparison of alternative optimization
% methods to the L-BFGS method that was ultimately employed in order to
% solve the problem. The archive containing its data will be labelled as
% "OptimMethods.dat"

% IMPORTANT!! The function "fmin_adam.m" will be needed to run this code 
% (but not for analyzing the results)

%% Part I.- Starting the Program and General Definitions
% Could be commented due to the large runtime of the algorithm
clear; clc; close all;

%% Part II.- Definition of Hyperparameters and Description of the Problem
% a.   Describe the dimensions of the problem
% Set the dimensions in the x-direction
xmin = -1;
xmax =  1;
% Set the maximum time dimension
tmax =  1;
% b.   Description of the Boundary Conditions (constraints in space)
% Prescribe the number of boundary conditions at either side (temporal)
tnum = 11;
numBoundaryConditionPoints = [tnum tnum];
% Set the value of x for the Boundary Conditions
x0BC1 = xmin*ones(1,numBoundaryConditionPoints(1));
x0BC2 = xmax*ones(1,numBoundaryConditionPoints(2));
% Set the value of t for the Boundary Conditions (provided a maximum time)
t0BC1 = linspace(0,tmax,numBoundaryConditionPoints(1));
t0BC2 = linspace(0,tmax,numBoundaryConditionPoints(2));
% Set the value of the function as zero at the BC (Dirichlet BCs)
u0BC1 = zeros(1,numBoundaryConditionPoints(1));
u0BC2 = zeros(1,numBoundaryConditionPoints(2));
% c.   Description of the Initial Conditions (constraints in time)
% Prescribe the number of initial condition points (spatial)
xnum = 21;
numInitialConditionPoints  = 21;
% Set the value of x for the Initial Conditions (linearized to [-1,1])
x0IC = linspace(xmin,xmax,numInitialConditionPoints);
% Set the value of t for the Initial Conditions
t0IC = zeros(1,numInitialConditionPoints);
% Set the value of the Initial Condition as a bell curve for this scenario
alpha = 5^2;
u0IC = exp(-alpha*x0IC.^2);
% d.   Generate the final constraints by group ICs and BCs
% Write all the constraints in vector form
XBC = [x0BC1 x0BC2];
TBC = [t0BC1 t0BC2];
UBC = [u0BC1 u0BC2];
XIC = x0IC;
TIC = t0IC;
UIC = u0IC;
X0 = [x0IC x0BC1 x0BC2];
T0 = [t0IC t0BC1 t0BC2];
U0 = [u0IC u0BC1 u0BC2];

% e.   Generate the full region of internal points by using collocation.
% Set the number of points to have beyond the initial ones.
% Describing the x-coordinate:
dt = (tmax-0)/(length(t0BC1)-1);
dx = (xmax-xmin)/(length(x0IC)-1);
Xc = (xmin):dx:(xmax);
Tc = (0):dt:(tmax);
% Describing the Time matrix TT and Space matrix XX
TT = []; XX = [];
for i = 1:length(Xc)
    TT = [TT Tc'];
end
for j = 1:length(Tc)
    XX = [XX; Xc];
end
% Describing the time vector tpts and space vector xpts
tpts = []; xpts = [];
for i = 1:length(Xc)
    for j = 1:length(Tc)
        tpts = [tpts, TT(j,i)];
        xpts = [xpts, XX(j,i)];
    end
end

%% Part II.- Define the Deep Learning Model
% Let us consider a multilayer perceptron architecture which has 2 fully
% connected operations and 20 hidden neurons.
numLayers = 2;
numNeurons = 20;

% The structure will be such that it requires 2 input channels (x and t)
% and delivers a single output channel u(x,t).
% Inserting all of these variables into a structure that can be passed
% around easily.
inputs = struct;
inputs.XBC = XBC;
inputs.TBC = TBC;
inputs.UBC = UBC;
inputs.XIC = XIC;
inputs.TIC = TIC;
inputs.UIC = UIC;
inputs.X0 = X0;
inputs.T0 = T0;
inputs.U0 = U0;
inputs.dataX = xpts';
inputs.dataT = tpts';

%% Part IV.- Evaluate and compare the characteristics of the model
% Let us first determine a set of particular time instances at which to
% study the model.
errIC = struct;
errBC = struct;
errFC = struct;
niter = struct;
neval = struct;
telap = struct;

%% Defining the options for each of the methods:
% Options for the fmincon LBFGS algorithm
options_a = optimoptions('fmincon', ... % Using the multivariate restricted optimization function fmincon
    'HessianApproximation','lbfgs', ... % Using the Lim-Memory BFGS algorithm for the Hessian
    'MaxIterations',1E4, ... % Needs to be sufficiently large so as to ensure convergence
    'MaxFunctionEvaluations',2E4, ... % Needs to be sufficiently large so as to ensure convergence
    'OptimalityTolerance',1e-5, ... % By default considering a tolerance of 0.00001
    'SpecifyObjectiveGradient',true); % User-defined gradient for the algorithm
% Options for the fmincon Conjugate Gradient
options_b = optimoptions('fmincon', ... % Using the multivariate restricted optimization function fmincon
    'SubproblemAlgorithm','cg',... % Using a conjugate gradient
    'HessianApproximation','finite-difference', ... % Using the Lim-Memory BFGS algorithm for the Hessian
    'MaxIterations',1E4, ... % Needs to be sufficiently large so as to ensure convergence
    'MaxFunctionEvaluations',2E4, ... % Needs to be sufficiently large so as to ensure convergence
    'OptimalityTolerance',1e-3, ... % By default considering a tolerance of 0.00001
    'SpecifyObjectiveGradient',true); % User-defined gradient for the algorithm 
% Options for the fminunc DFP
options_c = optimset('HessUpdate','dfp',.... % Using the DFP algorithm for the Hessian
    'MaxIter',1E4, ... % Needs to be sufficiently large so as to ensure convergence
    'MaxFunEvals',2E4,...
    'TolFun',1E-3); % Deliver the number of needed iterations 
% Options for the Nelsen-Mead Algorithm

% Options for the fsolve Levenberg Marquardt
options_e = optimoptions('fsolve', ... % Using the multivariate restricted optimization function fmincon
    'Algorithm','levenberg-marquardt',... % Using the Levenberg-Marquard
    'MaxIterations',1E4, ... % Needs to be sufficiently large so as to ensure convergence
    'MaxFunctionEvaluations',2E4, ... % Needs to be sufficiently large so as to ensure convergence
    'OptimalityTolerance',1e-3, ... % By default considering a tolerance 
    'FunctionTolerance',1e-3);  % Setting a tolerance for the loss function
% Options for the fminadam

%% Predefining some variables for the testing
tTest = 0.00;
numPredictions = 1001;
XTestA = linspace(-1,1,numPredictions);
TTestA = tTest*ones(1,numPredictions);
dlXTestA = dlarray(XTestA,'CB');
dlTTestA = dlarray(TTestA,'CB');
alpha = 5^2; k = 0.1;
UTestA = solveHeatEquation(XTestA,TTestA,alpha,k);
tTestB = linspace(-1,1,numPredictions);
xTestB = -1*ones(1,numPredictions);
dlXTestB = dlarray(xTestB,'CB');
dlTTestB = dlarray(tTestB,'CB');
xTestC =  1*ones(1,numPredictions);
dlXTestC = dlarray(xTestC,'CB');
dlTTestC = dlarray(tTestB,'CB');
tTest = 1.00;
numPredictions = 1001;
XTestD = linspace(-1,1,numPredictions);
TTestD = tTest*ones(1,numPredictions);
dlXTestD = dlarray(XTestD,'CB');
dlTTestD = dlarray(TTestD,'CB');
UTestD = solveHeatEquation(XTestD,TTestD,alpha,k);

%% Running the solutions

for i = 1:10
    parameters = generateParameters(numLayers,numNeurons);
% Option I
    tic;
    [parametersA,outputA] = PINN_HE_cstr(inputs,parameters,options_a,1);
    ttk = toc;
    dlUPred_A = model(parametersA,dlXTestA,dlTTestA,1);
    errIC.("case1_" + i) = mse(extractdata(dlUPred_A),UTestA);
    dlUPred_A = model(parametersA,dlXTestD,dlTTestD,1);
    errFC.("case1_" + i) = mse(extractdata(dlUPred_A),UTestD);
    dlUPred_A = model(parametersA,dlXTestB,dlTTestB,1);
    errBCL = mse(dlUPred_A,0*dlUPred_A);
    dlUPred_A = model(parametersA,dlXTestC,dlTTestC,1);
    errBCR = mse(dlUPred_A,0*dlUPred_A);
    errBC.("case1_" + i) = (errBCL+errBCR)/2;
    niter.("case1_" + i) = outputA.iterations;
    neval.("case1_" + i) = outputA.funcCount;
    telap.("case1_" + i) = ttk;
    str = "A" + i;
    disp(str)

% Option II
    tic;
    [parametersA,outputA] = PINN_HE_cstr(inputs,parameters,options_b,1);
    ttk = toc;
    dlUPred_A = model(parametersA,dlXTestA,dlTTestA,1);
    errIC.("case2_" + i) = mse(extractdata(dlUPred_A),UTestA);
    dlUPred_A = model(parametersA,dlXTestD,dlTTestD,1);
    errFC.("case2_" + i) = mse(extractdata(dlUPred_A),UTestD);
    dlUPred_A = model(parametersA,dlXTestB,dlTTestB,1);
    errBCL = mse(dlUPred_A,0*dlUPred_A);
    dlUPred_A = model(parametersA,dlXTestC,dlTTestC,1);
    errBCR = mse(dlUPred_A,0*dlUPred_A);
    errBC.("case2_" + i) = (errBCL+errBCR)/2;
    niter.("case2_" + i) = outputA.iterations;
    neval.("case2_" + i) = outputA.funcCount;
    telap.("case2_" + i) = ttk;
    str = "B" + i;
    disp(str)

% Option III
    tic;
    [parametersA,outputA] = PINN_HE_uncs(inputs,parameters,options_c,1);
    ttk = toc;
    dlUPred_A = model(parametersA,dlXTestA,dlTTestA,1);
    errIC.("case3_" + i) = mse(extractdata(dlUPred_A),UTestA);
    dlUPred_A = model(parametersA,dlXTestD,dlTTestD,1);
    errFC.("case3_" + i) = mse(extractdata(dlUPred_A),UTestD);
    dlUPred_A = model(parametersA,dlXTestB,dlTTestB,1);
    errBCL = mse(dlUPred_A,0*dlUPred_A);
    dlUPred_A = model(parametersA,dlXTestC,dlTTestC,1);
    errBCR = mse(dlUPred_A,0*dlUPred_A);
    errBC.("case3_" + i) = (errBCL+errBCR)/2;
    niter.("case3_" + i) = outputA.iterations;
    neval.("case3_" + i) = outputA.funcCount;
    telap.("case3_" + i) = ttk;
    str = "C" + i;
    disp(str)

% Option IV
    tic;
    [parametersA,outputA] = PINN_HE_srch(inputs,parameters,1);
    ttk = toc;
    dlUPred_A = model(parametersA,dlXTestA,dlTTestA,1);
    errIC.("case4_" + i) = mse(extractdata(dlUPred_A),UTestA);
    dlUPred_A = model(parametersA,dlXTestD,dlTTestD,1);
    errFC.("case4_" + i) = mse(extractdata(dlUPred_A),UTestD);
    dlUPred_A = model(parametersA,dlXTestB,dlTTestB,1);
    errBCL = mse(dlUPred_A,0*dlUPred_A);
    dlUPred_A = model(parametersA,dlXTestC,dlTTestC,1);
    errBCR = mse(dlUPred_A,0*dlUPred_A);
    errBC.("case4_" + i) = (errBCL+errBCR)/2;
    niter.("case4_" + i) = outputA.iterations;
    neval.("case4_" + i) = outputA.funcCount;
    telap.("case4_" + i) = ttk;
    str = "D" + i;
    disp(str)

% Option V
    tic;
    [parametersA,outputA] = PINN_HE_solve(inputs,parameters,options_e,1);
    ttk = toc;
    dlUPred_A = model(parametersA,dlXTestA,dlTTestA,1);
    errIC.("case5_" + i) = mse(extractdata(dlUPred_A),UTestA);
    dlUPred_A = model(parametersA,dlXTestD,dlTTestD,1);
    errFC.("case5_" + i) = mse(extractdata(dlUPred_A),UTestD);
    dlUPred_A = model(parametersA,dlXTestB,dlTTestB,1);
    errBCL = mse(dlUPred_A,0*dlUPred_A);
    dlUPred_A = model(parametersA,dlXTestC,dlTTestC,1);
    errBCR = mse(dlUPred_A,0*dlUPred_A);
    errBC.("case5_" + i) = (errBCL+errBCR)/2;
    niter.("case5_" + i) = outputA.iterations;
    neval.("case5_" + i) = outputA.funcCount;
    telap.("case5_" + i) = ttk;
    str = "E" + i;
    disp(str)

% Option VI
    tic;
    [parametersA,outputA] = PINN_HE_adam(inputs,parameters,1);
    ttk = toc;
    dlUPred_A = model(parametersA,dlXTestA,dlTTestA,1);
    errIC.("case6_" + i) = mse(extractdata(dlUPred_A),UTestA);
    dlUPred_A = model(parametersA,dlXTestD,dlTTestD,1);
    errFC.("case6_" + i) = mse(extractdata(dlUPred_A),UTestD);
    dlUPred_A = model(parametersA,dlXTestA,dlTTestA,1);
    errBCL = mse(dlUPred_A,0*dlUPred_A);
    dlUPred_A = model(parametersA,dlXTestA,dlTTestA,1);
    errBCR = mse(dlUPred_A,0*dlUPred_A);
    errBC.("case6_" + i) = (errBCL+errBCR)/2;
    niter.("case6_" + i) = outputA.iteration;
    neval.("case6_" + i) = outputA.funccount;
    telap.("case6_" + i) = ttk;
    str = "F" + i;
    disp(str)
end
%% Checkmark for results: 
% if the .mat data is loaded, the archive might be opened from here and
% only the sections above will be ran.

%% Storing the data obtained in an easier to handle matrix form:
errICss = []; errBCss = []; errFCss = []; niterss = []; nevalss = []; telapss = [];
errICs = [];  errBCs = [];  errFCs = [];  niters = [];  nevals = [];  telaps = [];
for i = 1:6
    for j = 1:10
        errICss(i,j) = errIC.("case" + i + "_" + j);
        errFCss(i,j) = errFC.("case" + i + "_" + j);
        errBCss(i,j) = errBC.("case" + i + "_" + j);
        niterss(i,j) = niter.("case" + i + "_" + j);
        nevalss(i,j) = neval.("case" + i + "_" + j);
        telapss(i,j) = telap.("case" + i + "_" + j);
    end
    errICs(i) = sum(errICss(i,:))/10;
    errFCs(i) = sum(errFCss(i,:))/10;
    errBCs(i) = sum(errBCss(i,:))/10;
    niters(i) = sum(niterss(i,:))/10;
    nevals(i) = sum(nevalss(i,:))/10;
    telaps(i) = sum(telapss(i,:))/10;
end

%% Plotting the accuracy of the methods
figure
subplot(1,2,1)
hold on
for i = 1:6
    scatter(errBCs(i), errICs(i), 1e2*ones(1,10),'filled','o')
end
hold off
title('Measurement of Accuracy','Interpreter','latex');
set(gca,'XScale','log'); set(gca,'YScale','log');
xlabel('Error in the Boundary Conditions','Interpreter','latex');
ylabel('Error in the Initial Conditions','Interpreter','latex');
grid minor
legend('Constrained L-BFGS','Constrained CG',...
    'Unconstrained DFP','Nelsen-Mead (Derivative-Free)',...
    'Levenberg-Marquard','Adam SGD',...
    'Interpreter','latex','Location','best');
box on

%% Plotting the efficiency of the methods
subplot(1,2,2)
hold on
for i = 1:6
    bubblechart(nevals(i), niters(i), telaps(i))
end
blgd = bubblelegend('Size of the ball in $t(s)$',...
    'Interpreter','latex','Location','northwest','Style','telescopic');
hold off
xlabel('Number of Function Evaluations','Interpreter','latex');
ylabel('Number of Iterations','Interpreter','latex');
grid minor
title('Measurement of Efficiency','Interpreter','latex');
set(gca,'XScale','log'); set(gca,'YScale','log');
xlim([5e1,5e4]); ylim([4e1,3e4]);
bubblelim([2,500])
% legend('Constrained L-BFGS','Constrained CG',...
%     'Unconstrained DFP','Nelsen-Mead (Derivative-Free)',...
%     'Levenberg-Marquard','Adam SGD',...
%     'Interpreter','latex','Location','best');
box on

%% Extra Functions
% 
% Solve Heat Function Equation
%
function u = solveHeatEquation(x,t,alpha,k)
% The functional form of the solution is dependent on these two values:
% \sum_i ai*sin(i pi x)*exp(-k*(i*pi)^2)
% Let us obtain each of the ai by Fourier transform
% \int_{-1}^1 exp(-alpha*x^2)*sin(i*pi*x) dx = ai * pi/2
    u = 1/2*integral(@(x) exp(-alpha*x.^2),0,1);
    for i = 1:100
        a = integral(@(x) exp(-alpha*x.^2).*cos(i*pi*x/2), 0, 1);
        % b = 2/pi*integral(@(x) exp(-alpha*x.^2).*sin(i*pi*x/2), -1, 1);
        u = u + a*exp(-k*(i*pi/2)^2.*t).*cos(i*pi.*x/2); % + b*exp(-k*(i*pi/2)^2.*t)*sin(i*pi.*x/2);
    end
end
%
% fmincon Objective Function
%
function [loss,gradientsV] = objectiveFunction(parametersV,dlX,dlT,dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,parameterNames,parameterSizes,af)
    % Rate of Decay
    a = 0.5;
    % Convert parameters to structure of dlarray objects.
    parametersV = dlarray(parametersV);
    parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
    % Evaluate model gradients and loss.
    [gradients,loss] = dlfeval(@modelGradients,parameters,dlX,dlT,dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,af);
    % Return loss and gradients for fmincon.
    gradientsV = parameterStructToVector(gradients);
    gradientsV = a*extractdata(gradientsV);
    loss = extractdata(loss);
end
%
% Generate the Model Gradients Function
%
function [gradients,loss] = modelGradients(parameters,dlX,dlT,dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,a)
    % Redefining dlX and dlT to include the ICs and BCs
    dlXX = [dlX, dlXIC, dlXBC];
    dlTT = [dlT, dlTIC, dlTBC];
    % Make predictions with the initial conditions.
    % U = model(parameters,dlX,dlT);
    UU = model(parameters,dlXX,dlTT,a);
    % Assume a value for the parameter k
    k = 0.1;
    % Calculate derivatives with respect to X and T.
    gradientsU = dlgradient(sum(UU,'all'),{dlXX,dlTT},'EnableHigherDerivatives',true); % +dlXBC+dlXIC//+dlTIC+dlTBC
    Ux = gradientsU{1};
    Ut = gradientsU{2};
    % Calculate second-order derivatives with respect to X.
    Uxx = dlgradient(sum(Ux,'all'),dlXX,'EnableHigherDerivatives',true); % +dlXBC+dlXIC
    % Calculate lossF. Enforce Burger's equation.
    f = Ut -k.*Uxx;
    zeroTarget = zeros(size(f), 'like', f);
    lossF = mse(f, zeroTarget);
    % Calculate lossI. Enforce initial conditions.
    dlUICPred = model(parameters,dlXIC,dlTIC,a);
    lossI = mse(dlUICPred, dlUIC);
    % Calculate lossB. Enforce boundary conditions. Using Neumann BCs
    dlUBCPred = model(parameters,dlXBC,dlTBC,a);
    Un = dlgradient(sum(dlUBCPred,'all'),dlXBC,'EnableHigherDerivatives',true);
    lossB = mse(dlUBCPred, dlUBC);
    % Combine losses.
    loss = lossF + lossI + lossB;
    % Calculate gradients with respect to the learnable parameters.
    gradients = dlgradient(loss,parameters);
end
%
% Generate the Model for the Full Network
%
function dlU = model(parameters,dlX,dlT,a)
    arguments
        parameters
        dlX
        dlT
        a = 1;
    end
    dlXT = [dlX;dlT];
    numLayers = numel(fieldnames(parameters))/2;
    % First fully connect operation.
    weights = parameters.fc1_Weights;
    if a > 1
        bias = 0*parameters.fc1_Bias;
    else
        bias = 1*parameters.fc1_Bias;
    end
    dlU = fullyconnect(dlXT,weights,bias);
    % tanh and fully connect operations for remaining layers.
    for i=2:numLayers
        name = "fc" + i;
        ep = 0.5; % Setting the parameter for this run 
        switch a
            case 1 % Hyperbolic Arctangent
                dlU = tanh(dlU);
            case 2 % Rectified Linear Unit
                dlU = max(0,dlU);
            case 3 % Sigmoid Activation Function
                dlU = 1./(1+exp(-dlU));
            case 4 % Exponential Rectified Linear Unit
                if dlU <= 0
                    dlU = 0.01*(exp(dlU)-1);
                end
            case 5 % Multiquadric RBF
                dlU = sqrt(1 + ep.^2*dlU.^2);
            case 6 % Gaussian RBF
                dlU = exp(-ep.^2*dlU.^2);
            
        end
        weights = parameters.(name + "_Weights");
        bias = parameters.(name + "_Bias")*0;
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
    if i ~= 0
        s = rng(i);
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
    parameters.fc1_Weights = initializeHe(sz,2);
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


function [parameters,output] = PINN_HE_cstr(in,parameters,options,af)
% Part III.- Train the Network Using fmincon
% Generate the vector with the parameters to be inserted in the algorithm.
[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters); 
parametersV = double(extractdata(parametersV));

% Generate the arrays with the data for loss function minimization
dlX = dlarray(in.dataX','CB');
dlT = dlarray(in.dataT','CB');
dlX0 = dlarray(in.X0,'CB');
dlT0 = dlarray(in.T0,'CB');
dlU0 = dlarray(in.U0,'CB');
dlXBC = dlarray(in.XBC,'CB');
dlTBC = dlarray(in.TBC,'CB');
dlUBC = dlarray(in.UBC,'CB');
dlXIC = dlarray(in.XIC,'CB');
dlTIC = dlarray(in.TIC,'CB');
dlUIC = dlarray(in.UIC,'CB');

% Define the objective function through the data obtained beforehand
objFun = @(parameters) objectiveFunction(parameters,dlX,dlT,...
    dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,parameterNames,parameterSizes,af);

% Run the fmincon algorithm and obtain the total runtime of the algorithm
tic
%[x,fval,exitflag,output,lambda,grad,hessian]
[parametersV,fval,exitflag,output,lambda,grad,hessian] = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
toc
% Revert the optimized version of the parameters to a struct. 
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
end


% Is this correct(?)
function [parameters,output] = PINN_HE_srch(in,parameters,af)
% Part III.- Train the Network Using fmincon
% Generate the vector with the parameters to be inserted in the algorithm.
[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters); 
parametersV = double(extractdata(parametersV));

% Generate the arrays with the data for loss function minimization
dlX = dlarray(in.dataX','CB');
dlT = dlarray(in.dataT','CB');
dlX0 = dlarray(in.X0,'CB');
dlT0 = dlarray(in.T0,'CB');
dlU0 = dlarray(in.U0,'CB');
dlXBC = dlarray(in.XBC,'CB');
dlTBC = dlarray(in.TBC,'CB');
dlUBC = dlarray(in.UBC,'CB');
dlXIC = dlarray(in.XIC,'CB');
dlTIC = dlarray(in.TIC,'CB');
dlUIC = dlarray(in.UIC,'CB');

% Define the objective function through the data obtained beforehand
objFun = @(parameters) objectiveFunction(parameters,dlX,dlT,...
    dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,parameterNames,parameterSizes,af);

% Run the fmincon algorithm and obtain the total runtime of the algorithm
tic
%[x,fval,exitflag,output,lambda,grad,hessian]
[parametersV,fval,exitflag,output] = fminsearch(objFun,parametersV);
toc
% Revert the optimized version of the parameters to a struct. 
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
end


function [parameters,output] = PINN_HE_uncs(in,parameters,options,af)
% Part III.- Train the Network Using fmincon
% Generate the vector with the parameters to be inserted in the algorithm.
[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters); 
parametersV = double(extractdata(parametersV));

% Generate the arrays with the data for loss function minimization
dlX = dlarray(in.dataX','CB');
dlT = dlarray(in.dataT','CB');
dlX0 = dlarray(in.X0,'CB');
dlT0 = dlarray(in.T0,'CB');
dlU0 = dlarray(in.U0,'CB');
dlXBC = dlarray(in.XBC,'CB');
dlTBC = dlarray(in.TBC,'CB');
dlUBC = dlarray(in.UBC,'CB');
dlXIC = dlarray(in.XIC,'CB');
dlTIC = dlarray(in.TIC,'CB');
dlUIC = dlarray(in.UIC,'CB');

% Define the objective function through the data obtained beforehand
objFun = @(parameters) objectiveFunction(parameters,dlX,dlT,...
    dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,parameterNames,parameterSizes,af);

% Run the fmincon algorithm and obtain the total runtime of the algorithm
tic
%[x,fval,exitflag,output,lambda,grad,hessian]
[parametersV,fval,exitflag,output] = fminsearch(objFun,parametersV,options);
toc
% Revert the optimized version of the parameters to a struct. 
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
end


function [parameters,output] = PINN_HE_adam(in,parameters,af)
% Part III.- Train the Network Using fmincon
% Generate the vector with the parameters to be inserted in the algorithm.
[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters); 
parametersV = double(extractdata(parametersV));

% Generate the arrays with the data for loss function minimization
dlX = dlarray(in.dataX','CB');
dlT = dlarray(in.dataT','CB');
dlX0 = dlarray(in.X0,'CB');
dlT0 = dlarray(in.T0,'CB');
dlU0 = dlarray(in.U0,'CB');
dlXBC = dlarray(in.XBC,'CB');
dlTBC = dlarray(in.TBC,'CB');
dlUBC = dlarray(in.UBC,'CB');
dlXIC = dlarray(in.XIC,'CB');
dlTIC = dlarray(in.TIC,'CB');
dlUIC = dlarray(in.UIC,'CB');

% Define the objective function through the data obtained beforehand
objFun = @(parameters) objectiveFunction(parameters,dlX,dlT,...
    dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,parameterNames,parameterSizes,af);

% Run the fmincon algorithm and obtain the total runtime of the algorithm
tic
%[x,fval,exitflag,output,lambda,grad,hessian]

[parametersV,fval,exitflag,output] = fmin_adam(objFun,parametersV); % ,1e-3, 0.9, 0.999, sqrt(eps), 10, options
toc
% Revert the optimized version of the parameters to a struct. 
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
end


function [parameters,output] = PINN_HE_solve(in,parameters,options,af)
% Part III.- Train the Network Using fmincon
% Generate the vector with the parameters to be inserted in the algorithm.
[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters); 
parametersV = double(extractdata(parametersV));

% Generate the arrays with the data for loss function minimization
dlX = dlarray(in.dataX','CB');
dlT = dlarray(in.dataT','CB');
dlX0 = dlarray(in.X0,'CB');
dlT0 = dlarray(in.T0,'CB');
dlU0 = dlarray(in.U0,'CB');
dlXBC = dlarray(in.XBC,'CB');
dlTBC = dlarray(in.TBC,'CB');
dlUBC = dlarray(in.UBC,'CB');
dlXIC = dlarray(in.XIC,'CB');
dlTIC = dlarray(in.TIC,'CB');
dlUIC = dlarray(in.UIC,'CB');

% Define the objective function through the data obtained beforehand
objFun = @(parameters) objectiveFunction(parameters,dlX,dlT,...
    dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,parameterNames,parameterSizes,af);

% Run the fmincon algorithm and obtain the total runtime of the algorithm
tic
%[x,fval,exitflag,output,lambda,grad,hessian]

[parametersV,fval,exitflag,output] = fsolve(objFun,parametersV,options);
toc
% Revert the optimized version of the parameters to a struct. 
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
end

