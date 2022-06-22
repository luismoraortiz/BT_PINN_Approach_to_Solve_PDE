%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Fourth Version of the Code                     %    by Luis Mora Ortiz 
%   Bachelor Thesis                                %    2021/2022          
%   A Deep Learning Approach to Solve Partial Differential Equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Considerations to be made:
%   The PDE of the problem is the Wave equation
%   The BCs are Dirichlet Boundary Conditions
%   The resolution algorithm uses Sigmoid activation function

% This problem will solve the 1DWE in order to compare four different types
% of meshing for the interior points to discuss the phenomena of
% overfitting and make an informed decision regarding other posterior
% problems. The archive containing all the data will be
% "Data_1DWE_Collocation.mat" 

%% Part I.- Starting the Program and General Definitions
% Could be commented due to the large runtime of the algorithm
clear; clc; close all;

% Let us first insert a section in which the variables to be changed are
% described beforehand, pertaining mainly to the number of layers, number
% of neurons, and the number of datapoints inserted in the algorithm:

%% Part II.- Definition of Hyperparameters and Description of the Problem
% a. Let us first provide the key hyperparameters and geometrical defs.
% Prescribe the number of boundary conditions at either side (temporal)
tnum = 21;
% Prescribe the number of initial condition points (spatial)
xnum = 41;
% Set the number of points to have beyond the initial ones.
numICPs = 1e3;
% Let us consider a multilayer perceptron architecture which has 9 fully
% connected operations and 20 hidden neurons.
numLayers = 15;
numNeurons = 10;
% Also as auxiliary variables, the following constraints of fmincon.
% Set the maximum number of iterations before convergence.
numiter = 2000;
% Set the maximum number of function evaluations before convergence.
numevs = 2000;
% Set the maximum level of tolerance under which a solution is accepted.
maxtol = 1e-3;
% Describe the dimensions of the problem
% Set the dimensions in the x-direction
xmin = -1;
xmax =  1;
% Set the maximum time dimension
tmax =  1;

% b. Description of the Boundary Conditions (constraints in space)
% Prescribe the number of boundary conditions at either side (temporal)
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

% c. Description of the Initial Conditions (constraints in time)
% Prescribe the number of initial condition points (spatial)
numInitialConditionPoints  = xnum;
% Set the value of x for the Initial Conditions (linearized to [-1,1])
x0IC = linspace(xmin,xmax,numInitialConditionPoints);
% Set the value of t for the Initial Conditions
t0IC = zeros(1,numInitialConditionPoints);
% Set the value of the Initial Condition as a bell curve for this scenario
alpha = 5^2;
u0IC = exp(-alpha*x0IC.^2);

% d. Generate the final constraints by group ICs and BCs
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

%% e.- Define the Deep Learning Model
% The structure will be such that it requires 2 input channels (x and t)
% and delivers a single output channel u(x,t).
% This will be predefined as the boundary conditions will not be modified
% in any of the scenarios proposed.
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

% d.   Select the options for the fmincon algorithm. Could add the
% iter-detailed variable to get further insight into the results.
options = optimoptions('fmincon', ... % Using the multivariate restricted optimization function fmincon
    'HessianApproximation','lbfgs', ... % Using the Lim-Memory BFGS algorithm for the Hessian
    'MaxIterations',numiter, ... % Needs to be sufficiently large so as to ensure convergence
    'MaxFunctionEvaluations',numevs, ... % Needs to be sufficiently large so as to ensure convergence
    'OptimalityTolerance',maxtol, ... % By default considering a tolerance of 0.00001
    'SpecifyObjectiveGradient',true); % User-defined gradient for the algorithm

%% Part III.- Solve the problem for comparison
% Let us first determine a set of particular time instances at which to
% study the model, using the alternative approach.
numPredictions = 1001;
tTestInit = 0.00;
TTestInit = tTestInit*ones(1,numPredictions);
tTestFinal = 1.00;
TTestFinal = tTestFinal*ones(1,numPredictions);
XTest = linspace(-1,1,numPredictions);
dlXTest = dlarray(XTest,'CB');
dlTTestInit = dlarray(TTestInit,'CB');
dlTTestFinal = dlarray(TTestFinal,'CB');
dlUPred = struct;

XLeft = -1*ones(1,numPredictions);
TSides = linspace(0,1,numPredictions);
XRight = 1*ones(1,numPredictions);
dlXLeft = dlarray(XLeft,'CB');
dlTSides = dlarray(TSides,'CB');
dlXRight = dlarray(XRight,'CB');

% The expected solution will be recalled from the solveHeatEquation
% program, which develops a Heat Equation:
u_exp_0 = solveWaveEquation(XTest,tTestInit,5^2,0.01);
u_exp_1 = solveWaveEquation(XTest,tTestFinal,5^2,0.01);

% Developping a struct for the predictions
error_model_A = struct; error_model_B = struct;
error_model_C = struct; error_model_D = struct;

% Developping a struct for the outputs and a matrix of exit flags
outputs_all = struct;
flags_A = []; flags_B = []; flags_C = []; flags_D = []; ttk = struct;

% Initializing the seeds
seed = 0;

% e.   Generate the full region of internal points by using collocation.
% This will be accomplished by a set of 4 different means, which will be
% solved individually.

%%   A. True Random Meshing: randomly selecting n points within the range
% Set the number of points to have beyond the initial ones.
numInternalCollocationPoints = numICPs;
% Implementing the loop
for randomICPs = 0:99
    % Describing the random data internal points
    points = rand(numInternalCollocationPoints,2);
    dataX = xmin + (xmax-xmin)*points(:,1);
    dataT = tmax*points(:,2);
    inputs.dataX = dataX;
    inputs.dataT = dataT;
    % Implementing the solution of the model
    % Generate the number of parameters
    [parameters,seed] = generateParameters(numLayers,numNeurons,seed);
    % Implement the training of the model for the parameters
    tic
    [parameters,outputs,exitflag] = PINN_HE_Dirichlet(inputs,parameters,options);
    ttk.("A" + randomICPs) = toc;
    outputs_all.("A" + randomICPs) = outputs;
    flags_A = [flags_A,exitflag];
    % Solving for the model according to the testing
    dlUU = model(parameters,dlXTest,dlTTestInit);
    dlUPred_ICPs_A0.("sol" + randomICPs) = dlUU;
    % Obtaining the associated mean squared error
    error_model_A.("nICP_0_" + randomICPs) = mse(dlUU,u_exp_0);
    % Solving for the model according to the testing
    dlUU = model(parameters,dlXTest,dlTTestFinal);
    dlUPred_ICPs_A1.("sol" + randomICPs) = dlUU;
    % Obtaining the associated mean squared error
    error_model_A.("nICP_1_" + randomICPs) = mse(dlUU,u_exp_1);
    disp("numICPs " + randomICPs);
    % Solving the model on the left
    dlUU = model(parameters,dlXLeft,dlTSides);
    error_model_A.("nICP_L_" + randomICPs) = mse(dlUU,0*dlUU);
    dlUPred_ICPs_Left.("sol" + randomICPs) = dlUU;
    dlUU = model(parameters,dlXRight,dlTSides);
    error_model_A.("nICP_R_" + randomICPs) = mse(dlUU,0*dlUU);
    dlUPred_ICPs_Right.("sol" + randomICPs) = dlUU;
end

%% B. Linear Collocation Meshing: using a linear space for a rect. lattice.
% Describing the x-coordinate and t-coordinate:
dt = (tmax-0)/(length(t0BC1)-1);
dx = (xmax-xmin)/(length(x0IC)-1);
% Implementing the colon operator
Xc = (xmin+dx):dx:(xmax-dx);
Tc = (0+dt):dt:(tmax-dt);
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
% Inputting the data on the inputs.
    inputs.dataX = xpts';
    inputs.dataT = tpts';
% Implementing the loop
for randomICPs = 0:99
    % Implementing the solution of the model
    % Generate the number of parameters
    [parameters,seed] = generateParameters(numLayers,numNeurons,seed);
    % Implement the training of the model for the parameters
    [parameters,outputs,exitflag] = PINN_HE_Dirichlet(inputs,parameters,options);
    outputs_all.("B" + randomICPs) = outputs;
    flags_B = [flags_B,exitflag];
    % Solving for the model according to the testing
    dlUU = model(parameters,dlXTest,dlTTestInit);
    dlUPred_ICPs_B0.("sol" + randomICPs) = dlUU;
    % Obtaining the associated mean squared error
    error_model_B.("nICP_0_" + randomICPs) = mse(dlUU,u_exp_0);
    % Solving for the model according to the testing
    dlUU = model(parameters,dlXTest,dlTTestFinal);
    dlUPred_ICPs_B1.("sol" + randomICPs) = dlUU;
    % Obtaining the associated mean squared error
    error_model_B.("nICP_1_" + randomICPs) = mse(dlUU,u_exp_1);
    disp("numICPs " + randomICPs);
end

%% C. Collocation with more weighing to the sides
% Describing the x-coordinate and t-coordinate:
dt1 = (tmax-0)/(length(t0BC1)-1); 
dx1 = (xmax-xmin)/(length(x0IC)-1);
dt2 = dt1/20; dx2 = dx1/20;
% Implementing the colon operator
Xc = [(xmin+dx2):dx2:(xmin+dx1-dx2), ...
    (xmax+dx1):dx1:(xmin-dx1),...
    (xmax-dx1+dx2):dx2:(xmax-dx2)];
Tc = [(0+dt2):dt2:(0.1*tmax-dt2), ...
    (0.1*tmax):dt1:(tmax-dt1)];
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
% Inputting the data on the inputs.
    inputs.dataX = xpts';
    inputs.dataT = tpts';
% Implementing the loop
for randomICPs = 0:99
    % Implementing the solution of the model
    % Generate the number of parameters
    [parameters,seed] = generateParameters(numLayers,numNeurons,seed);
    % Implement the training of the model for the parameters
    [parameters,outputs,exitflag] = PINN_HE_Dirichlet(inputs,parameters,options);
    outputs_all.("C" + randomICPs) = outputs;
    flags_C = [flags_C,exitflag];
    % Solving for the model according to the testing
    dlUU = model(parameters,dlXTest,dlTTestInit);
    dlUPred_ICPs_C0.("sol" + randomICPs) = dlUU;
    % Obtaining the associated mean squared error
    error_model_C.("nICP_0_" + randomICPs) = mse(dlUU,u_exp_0);
    % Solving for the model according to the testing
    dlUU = model(parameters,dlXTest,dlTTestFinal);
    dlUPred_ICPs_C1.("sol" + randomICPs) = dlUU;
    % Obtaining the associated mean squared error
    error_model_C.("nICP_1_" + randomICPs) = mse(dlUU,u_exp_1);
    disp("numICPs " + randomICPs);
end

%% D. Min. Distance Random Meshing: selecting n random pts with min dist
% Set the number of points to have beyond the initial ones.
numInternalCollocationPoints = numICPs;
% Define the minimum distance between adjacent points 
dist = 5e-4; 
% This could lead to an error depending on the ratio between the size of
% the grid and the number of chosen points.
% Predefining the data arrays
points = rand(1,2);
dataX = xmin + (xmax-xmin)*points(:,1);
dataT = tmax*points(:,2);
% Describing the random data internal points
cont = 1;

for i = 1:numInternalCollocationPoints
    % Generate the first random point
    while cont
        pti = rand(1,2);
        [~, d] = knnsearch(pti,points);
        if min(d) > dist
            dataX = [dataX; xmin + (xmax-xmin)*pti(:,1)];
            dataT = [dataT; tmax*pti(:,2)];
            cont = 0;
            points = [points; pti];
        end
    end
    cont = 1;
end
points = rand(numInternalCollocationPoints,2);

% Inserting the datasets to the objective structure
inputs.dataX = dataX;
inputs.dataT = dataT;
% Implementing the loop
for randomICPs = 0:99
    % Implementing the solution of the model
    % Generate the number of parameters
    [parameters,seed] = generateParameters(numLayers,numNeurons,seed);
    % Implement the training of the model for the parameters
    [parameters,outputs,exitflag] = PINN_HE_Dirichlet(inputs,parameters,options);
    outputs_all.("D" + randomICPs) = outputs;
    flags_D = [flags_D,exitflag];
    % Solving for the model according to the testing
    dlUU = model(parameters,dlXTest,dlTTestInit);
    dlUPred_ICPs_D0.("sol" + randomICPs) = dlUU;
    % Obtaining the associated mean squared error
    error_model_D.("nICP_0_" + randomICPs) = mse(dlUU,u_exp_0);
    % Solving for the model according to the testing
    dlUU = model(parameters,dlXTest,dlTTestFinal);
    dlUPred_ICPs_D1.("sol" + randomICPs) = dlUU;
    % Obtaining the associated mean squared error
    error_model_D.("nICP_1_" + randomICPs) = mse(dlUU,u_exp_1);
    disp("numICPs " + randomICPs);
end

%% Evaluate and compare the characteristics of the model
for i = 1:100
    S = i-1;
    randomSweepError_A(1,i) = error_model_A.("nICP_0_" + S);
    randomSweepError_A(2,i) = error_model_A.("nICP_1_" + S);
    randomSweepError_B(1,i) = error_model_B.("nICP_0_" + S);
    randomSweepError_B(2,i) = error_model_B.("nICP_1_" + S);
    randomSweepError_C(1,i) = error_model_C.("nICP_0_" + S);
    randomSweepError_C(2,i) = error_model_C.("nICP_1_" + S);
    randomSweepError_D(1,i) = error_model_D.("nICP_0_" + S);
    randomSweepError_D(2,i) = error_model_D.("nICP_1_" + S);
end

%% Checkmark for results: 
% if the .mat data is loaded, the archive might be opened from here and
% only the sections above will be ran.
                  
%% Graphing the Results:
% Modelling the initial conditions of the problem
% Displaying the error with random meshing
figure
clf
eff = extractdata(sum(randomSweepError_A(1,:)<0.01));
str = 'Effectiveness = ' + string(eff) + '/100)';
sgtitle('Random Meshing Network. (' + str,'Interpreter','latex');
for i = 1:100
    subplot(10,10,i)
    S = i-1;
    grid on
    hold on
    plot(XTest,u_exp_0,'b--');
    if randomSweepError_A(1,i)<0.01
        plot(XTest,extractdata(dlUPred_ICPs_A0.("sol" + S)),'g')
    else
        plot(XTest,extractdata(dlUPred_ICPs_A0.("sol" + S)),'r')
    end
    hold off
end
% Displaying the error with rectangular lattice meshing
figure
clf
eff = extractdata(sum(randomSweepError_B(1,:)<0.01));
str = 'Effectiveness = ' + string(eff) + '/100)';
sgtitle('Collocation Meshing Network. (' + str,'Interpreter','latex');
for i = 1:100
    subplot(10,10,i)
    S = i-1;
    grid on
    hold on
    plot(XTest,u_exp_0,'b--');
    if randomSweepError_B(1,i)<0.01
        plot(XTest,extractdata(dlUPred_ICPs_B0.("sol" + S)),'g')
    else
        plot(XTest,extractdata(dlUPred_ICPs_B0.("sol" + S)),'r')
    end
    hold off
end
% Displaying the error with side weighing meshing
figure
clf
eff = extractdata(sum(randomSweepError_C(1,:)<0.01));
str = 'Effectiveness = ' + string(eff) + '/100)';
sgtitle('Non-Linear Meshing Network. (' + str,'Interpreter','latex');
for i = 1:100
    subplot(10,10,i)
    S = i-1;
    grid on
    hold on
    plot(XTest,u_exp_0,'b--');
    if randomSweepError_C(1,i)<0.01
        plot(XTest,extractdata(dlUPred_ICPs_C0.("sol" + S)),'g')
    else
        plot(XTest,extractdata(dlUPred_ICPs_C0.("sol" + S)),'r')
    end
    hold off
end
% Displaying the error with random meshing with min. distance
figure
clf
eff = extractdata(sum(randomSweepError_D(1,:)<0.01));
str = 'Effectiveness = ' + string(eff) + '/100)';
sgtitle('Minimum Distance Random Meshing Network. (' + str,'Interpreter','latex');
for i = 1:100
    subplot(10,10,i)
    S = i-1;
    grid on
    hold on
    plot(XTest,u_exp_0,'b--');
    if randomSweepError_D(1,i)<0.01
        plot(XTest,extractdata(dlUPred_ICPs_D0.("sol" + S)),'g')
    else
        plot(XTest,extractdata(dlUPred_ICPs_D0.("sol" + S)),'r')
    end
    hold off
end

%% This models the behavior of the problem at the final time t = 1s
% Displaying the error with random meshing
figure
clf
eff = extractdata(sum(randomSweepError_A(2,:)<0.001));
str = 'Effectiveness = ' + string(eff) + '/100)';
sgtitle('Random Meshing Network. (' + str,'Interpreter','latex');
for i = 1:100
    subplot(10,10,i)
    S = i-1;
    grid on
    hold on
    plot(XTest,u_exp_1,'b--');
    if randomSweepError_A(2,i)<0.001
        plot(XTest,extractdata(dlUPred_ICPs_A1.("sol" + S)),'g')
    else
        plot(XTest,extractdata(dlUPred_ICPs_A1.("sol" + S)),'r')
    end
    hold off
end
% Displaying the error with rectangular lattice meshing
figure
clf
eff = extractdata(sum(randomSweepError_B(2,:)<0.001));
str = 'Effectiveness = ' + string(eff) + '/100)';
sgtitle('Collocation Meshing Network. (' + str,'Interpreter','latex');
for i = 1:100
    subplot(10,10,i)
    S = i-1;
    grid on
    hold on
    plot(XTest,u_exp_1,'b--');
    if randomSweepError_B(2,i)<0.001
        plot(XTest,extractdata(dlUPred_ICPs_B1.("sol" + S)),'g')
    else
        plot(XTest,extractdata(dlUPred_ICPs_B1.("sol" + S)),'r')
    end
    hold off
end
% Displaying the error with side weighing meshing
figure
clf
eff = extractdata(sum(randomSweepError_C(2,:)<0.005));
str = 'Effectiveness = ' + string(eff) + '/100)';
sgtitle('Non-Linear Meshing Network. (' + str,'Interpreter','latex');
for i = 1:100
    subplot(10,10,i)
    S = i-1;
    grid on
    hold on
    plot(XTest,u_exp_1,'b--');
    if randomSweepError_C(2,i)<0.005
        plot(XTest,extractdata(dlUPred_ICPs_C1.("sol" + S)),'g')
    else
        plot(XTest,extractdata(dlUPred_ICPs_C1.("sol" + S)),'r')
    end
    hold off
end
% Displaying the error with random meshing with min. distance
figure
clf
eff = extractdata(sum(randomSweepError_D(2,:)<0.001));
str = 'Effectiveness = ' + string(eff) + '/100)';
sgtitle('Minimum Distance Random Meshing Network. (' + str,'Interpreter','latex');
for i = 1:100
    subplot(10,10,i)
    S = i-1;
    grid on
    hold on
    plot(XTest,u_exp_1,'b--');
    if randomSweepError_D(2,i)<0.001
        plot(XTest,extractdata(dlUPred_ICPs_D1.("sol" + S)),'g')
    else
        plot(XTest,extractdata(dlUPred_ICPs_D1.("sol" + S)),'r')
    end
    hold off
end

%% Plotting the histogram with the normalized measurements of the error
% This will be a relative Linfty error
rSE(1,1) = sum(randomSweepError_A(1,:))/(100*max(u_exp_0)^2);
rSE(1,2) = sum(randomSweepError_B(1,:))/(100*max(u_exp_0)^2);
rSE(1,3) = sum(randomSweepError_C(1,:))/(100*max(u_exp_0)^2);
rSE(1,4) = sum(randomSweepError_D(1,:))/(100*max(u_exp_0)^2);
rSE(2,1) = sum(randomSweepError_A(2,:))/(100*max(u_exp_1)^2);
rSE(2,2) = sum(randomSweepError_B(2,:))/(100*max(u_exp_1)^2);
rSE(2,3) = sum(randomSweepError_C(2,:))/(100*max(u_exp_1)^2);
rSE(2,4) = sum(randomSweepError_D(2,:))/(100*max(u_exp_1)^2);
% rSE = extractdata(rSE);
% There might occur an error saying that the datatype is not adequate, for
% which the line above will be employed
figure
X = categorical({'Random Meshing',...
    'Rectangular Lattice',...
    'Reinforced Lattice',...
    'Min. Distance Random Meshing'});
X = reordercats(X,{'Random Meshing',...
    'Rectangular Lattice',...
    'Reinforced Lattice',...
    'Min. Distance Random Meshing'});
chars = [];
bar(X,rSE)
grid minor
ylabel('Relative Error at selected times','Interpreter','latex');
legend('Initial Time ($t=0$)','Final Time ($t=1$)',...
    'Interpreter','latex','Location','best');
set(gca,'YScale','log')  
ylim([1e-3,1e0])

%% Comparing some successful and unsuccessful solutions
k = mse(u_exp_0);
figure
xx = []; yy = []; 
for i = 0:99
    xx(i+1) = outputs_all.("A" + i).iterations;
    yy(i+1) = error_model_A.("nICP_0_" + i)/k;
end
subplot(2,2,1)
hold on
scatter(xx(yy<2e-2),yy(yy<2e-2),'bo');
scatter(xx(yy>2e-2),yy(yy>2e-2),'rx');
hold off
set(gca,'YScale','log'); set(gca,'XScale','log');
ylabel('Error in the Initial Conditions', 'Interpreter','latex');
xlabel('Number of iterations','Interpreter','latex');
eff = sum(yy<2e-2);
str = "Effectiveness = " + string(eff) + "/100";
title("Random Meshing with " + str,'Interpreter','latex')
grid minor
for i = 0:99
    xx(i+1) = outputs_all.("B" + i).iterations;
    yy(i+1) = error_model_B.("nICP_0_" + i)/k;
end
box on
subplot(2,2,2)
hold on
scatter(xx(yy<2e-2),yy(yy<2e-2),'bo');
scatter(xx(yy>2e-2),yy(yy>2e-2),'rx');
hold off
set(gca,'YScale','log'); set(gca,'XScale','log');
ylabel('Error in the Initial Conditions', 'Interpreter','latex');
xlabel('Number of iterations','Interpreter','latex');
eff = sum(yy<2e-2);
str = "Effectiveness = " + string(eff) + "/100";
title("Rectangular Lattice with " + str,'Interpreter','latex')
grid minor
for i = 0:99
    xx(i+1) = outputs_all.("C" + i).iterations;
    yy(i+1) = error_model_C.("nICP_0_" + i)/k;
end
box on
subplot(2,2,3)
hold on
scatter(xx(yy<2e-2),yy(yy<2e-2),'bo');
scatter(xx(yy>2e-2),yy(yy>2e-2),'rx');
hold off
set(gca,'YScale','log'); set(gca,'XScale','log');
ylabel('Error in the Initial Conditions', 'Interpreter','latex');
xlabel('Number of iterations','Interpreter','latex');
eff = sum(yy<2e-2);
str = "Effectiveness = " + string(eff) + "/100";
title("Reinforced Lattice with " + str,'Interpreter','latex')
grid minor
for i = 0:99
    xx(i+1) = outputs_all.("D" + i).iterations;
    yy(i+1) = error_model_D.("nICP_0_" + i)/k;
end
box on
subplot(2,2,4)
hold on
scatter(xx(yy<2e-2),yy(yy<2e-2),'bo');
scatter(xx(yy>2e-2),yy(yy>2e-2),'rx');
hold off
set(gca,'YScale','log'); set(gca,'XScale','log');
ylabel('Error in the Initial Conditions', 'Interpreter','latex');
xlabel('Number of iterations','Interpreter','latex');
eff = sum(yy<2e-2);
str = "Effectiveness = " + string(eff) + "/100";
title("Min. Dist. Random Meshing with " + str,'Interpreter','latex')
grid minor
box on

%% The same but for final conditions to talk overfitting
k = mse(u_exp_1);
figure
xx = []; yy = []; 
for i = 0:99
    xx(i+1) = outputs_all.("A" + i).iterations;
    yy(i+1) = error_model_A.("nICP_1_" + i)/k;
end
subplot(2,2,1)
hold on
scatter(xx(yy<2e-2),yy(yy<2e-2),'bo');
scatter(xx(yy>2e-2),yy(yy>2e-2),'rx');
hold off
set(gca,'YScale','log'); set(gca,'XScale','log');
ylabel('Error in the Initial Conditions', 'Interpreter','latex');
xlabel('Number of iterations','Interpreter','latex');
eff = sum(yy<2e-2);
str = "Effectiveness = " + string(eff) + "/100";
title("Random Meshing with " + str,'Interpreter','latex')
grid minor
for i = 0:99
    xx(i+1) = outputs_all.("B" + i).iterations;
    yy(i+1) = error_model_B.("nICP_1_" + i)/k;
end
box on
subplot(2,2,2)
hold on
scatter(xx(yy<2e-2),yy(yy<2e-2),'bo');
scatter(xx(yy>2e-2),yy(yy>2e-2),'rx');
hold off
set(gca,'YScale','log'); set(gca,'XScale','log');
ylabel('Error in the Initial Conditions', 'Interpreter','latex');
xlabel('Number of iterations','Interpreter','latex');
eff = sum(yy<2e-2);
str = "Effectiveness = " + string(eff) + "/100";
title("Rectangular Lattice with " + str,'Interpreter','latex')
grid minor
for i = 0:99
    xx(i+1) = outputs_all.("C" + i).iterations;
    yy(i+1) = error_model_C.("nICP_1_" + i)/k;
end
box on
subplot(2,2,3)
hold on
scatter(xx(yy<2e-2),yy(yy<2e-2),'bo');
scatter(xx(yy>2e-2),yy(yy>2e-2),'rx');
hold off
set(gca,'YScale','log'); set(gca,'XScale','log');
ylabel('Error in the Initial Conditions', 'Interpreter','latex');
xlabel('Number of iterations','Interpreter','latex');
eff = sum(yy<2e-2);
str = "Effectiveness = " + string(eff) + "/100";
title("Reinforced Lattice with " + str,'Interpreter','latex')
grid minor
for i = 0:99
    xx(i+1) = outputs_all.("D" + i).iterations;
    yy(i+1) = error_model_D.("nICP_1_" + i)/k;
end
box on
subplot(2,2,4)
hold on
scatter(xx(yy<2e-2),yy(yy<2e-2),'bo');
scatter(xx(yy>2e-2),yy(yy>2e-2),'rx');
hold off
set(gca,'YScale','log'); set(gca,'XScale','log');
ylabel('Error in the Initial Conditions', 'Interpreter','latex');
xlabel('Number of iterations','Interpreter','latex');
eff = sum(yy<2e-2);
str = "Effectiveness = " + string(eff) + "/100";
title("Min. Dist. Random Meshing with " + str,'Interpreter','latex')
grid minor
box on

%% Considering the accuracy of the random meshing
ttaken = zeros(1,100);
niter = zeros(1,100);
feval = zeros(1,100);
errIC = zeros(1,100);
errBC = zeros(1,100);
for i = 0:99
    ttaken(i+1) = ttk.("A" + i);
    niter(i+1) = outputs_all.("A" + i).iterations;
    feval(i+1) = outputs_all.("A" + i).funcCount;
    errIC(i+1) = error_model_A.("nICP_0_" + i);
    errBC(i+1) = (error_model_A.("nICP_L_" + i)+error_model_A.("nICP_R_" + i))/2;
end

%% Plots contained in the MATLAB
cc = (errIC<0.04)&(errBC<0.04);
figure
hold on
scatter(errIC(cc),errBC(cc),'bo')
scatter(errIC(~cc),errBC(~cc),'rx')
hold off
vv = sum(cc);
str = "Number of Compliant Trials: " + vv + "/100";
title(str,'Interpreter','latex');
xlabel('Error in the Initial Conditions','Interpreter','latex');
ylabel('Error in the Boundary Conditions','Interpreter','latex');
grid minor
set(gca,'XScale','log'); set(gca,'YScale','log');
box on

figure
bubblechart(niter,feval,ttaken);
blgd = bubblelegend('Size of the ball in $t(s)$',...
    'Interpreter','latex','Location','northwest');
xlabel('Number of Function Evaluations','Interpreter','latex');
ylabel('Number of Iterations','Interpreter','latex');
set(gca,'XScale','log'); set(gca,'YScale','log');
blgd = bubblelegend('Size of the ball in $t(s)$',...
    'Interpreter','latex','Location','northwest');
grid minor
bubblesize([3,25])
bubblelim([1.6,1e3])
box on

%%
mean(errBC(cc))
mean(errIC(cc))
var(errBC(cc))
var(errIC(cc))
cov(errIC(cc)',errBC(cc)')
corr(errBC(cc)',errIC(cc)')

min(ttaken(cc)')
max(ttaken(cc)')
mean(ttaken(cc)')
mean(feval(cc)')
mean(niter(cc)')
var(ttaken(cc)')
var(feval(cc)')
var(niter(cc)')
corr(feval(cc)',ttaken(cc)')
corr(feval(cc)',niter(cc)')
corr(ttaken(cc)',niter(cc)')

%%
% Choosing 22 as an example of invalid solution and 23 as an example of
% a valid one
% ii = 58; jj = 69;
% figure
% subplot(2,2,1)
% hold on
% plot(extractdata(dlXTest),extractdata(dlUPred_ICPs_A0.("sol"+ii)),'r')
% plot(extractdata(dlXTest),extractdata(dlUPred_ICPs_A0.("sol"+jj)),'b')
% plot(extractdata(dlXTest),u_exp_0,'k--')
% hold off
% title('Initial Condition $u(x,t = 0)$','Interpreter','latex');
% xlabel('x-coordinate','Interpreter','latex');
% ylabel('Value of $u$','Interpreter','latex');
% grid minor;
% box on
% subplot(2,2,2)
% hold on
% plot(extractdata(dlXTest),extractdata(dlUPred_ICPs_A1.("sol"+ii)),'r')
% plot(extractdata(dlXTest),extractdata(dlUPred_ICPs_A1.("sol"+jj)),'b')
% plot(extractdata(dlXTest),u_exp_1,'k--')
% hold off
% title('Final Solution $u(x,t = 1)$','Interpreter','latex');
% xlabel('x-coordinate','Interpreter','latex');
% ylabel('Value of $u$','Interpreter','latex');
% grid minor;
% box on
% subplot(2,2,3)
% hold on
%plot(extractdata(dlTSides),extractdata(dlUPred_ICPs_Left.("sol" + ii)),'r')
% plot(extractdata(dlTSides),extractdata(dlULeft),'r')
%plot(extractdata(dlTSides),extractdata(dlUPred_ICPs_Left.("sol" + jj)),'b')
% plot(extractdata(dlTSides),extractdata(dlULeft),'b')
% plot(extractdata(dlTSides),extractdata(dlTSides)*0,'k--')
% hold off
% title('Left Boundary Condition $u(x = -1,t)$','Interpreter','latex');
% xlabel('t-coordinate','Interpreter','latex');
% ylabel('Value of $u$','Interpreter','latex');
% grid minor;
% box on
% subplot(2,2,4)
% hold on
%plot(extractdata(dlTSides),extractdata(dlUPred_ICPs_Right.("sol" + ii)),'r')
% plot(extractdata(dlTSides),extractdata(dlURight),'r')
%plot(extractdata(dlTSides),extractdata(dlUPred_ICPs_Right.("sol" + jj)),'b')
% plot(extractdata(dlTSides),extractdata(dlURight),'b')
% plot(extractdata(dlTSides),extractdata(dlTSides)*0,'k--')
% hold off
% title('Right Boundary Condition $u(x = +1,t)$','Interpreter','latex');
% xlabel('t-coordinate','Interpreter','latex');
% ylabel('Value of $u$','Interpreter','latex');
% grid minor;
% legend('Unsuccessful Trial','Successful Trial','Expected Behavior',...
%     'Location','best','Interpreter','latex')
% box on

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
        u = u + a*exp(-k*(i*pi/2)^2.*t)*cos(i*pi.*x/2); % + b*exp(-k*(i*pi/2)^2.*t)*sin(i*pi.*x/2);
    end
end

% The Fourier expansion approach that was used in the HE has not been
% reproduced in the context of this code.
function u = solveWaveEquation(x,t,alpha,c)
% The functional form of the solution is dependent on these two values:
% \sum_i ai*sin(i pi x)*exp(-k*(i*pi)^2)
% Let us obtain each of the ai by Fourier transform
% \int_1^1 exp(-alpha*x^2)*sin(i*pi*x) dx = ai * pi/2
c = sqrt(c);
u = 1/2*integral(@(x) exp(-alpha*x.^2),0,1);
for i = 1:100
    a = integral(@(x) exp(-alpha*x.^2).*cos(i*pi*x/2), 0, 1);
    u = u + a*cos(c*i*pi/2.*t)*cos(i*pi/2.*x);
end
end
%
% fmincon Objective Function
%
function [loss,gradientsV] = objectiveFunction(parametersV,dlX,dlT,dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,parameterNames,parameterSizes)
    % Rate of Decay
    a = 0.5;
    % Convert parameters to structure of dlarray objects.
    parametersV = dlarray(parametersV);
    parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
    % Evaluate model gradients and loss.
    [gradients,loss] = dlfeval(@modelGradients,parameters,dlX,dlT,dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC);
    % Return loss and gradients for fmincon.
    gradientsV = parameterStructToVector(gradients);
    gradientsV = a*extractdata(gradientsV);
    loss = extractdata(loss);
end
%
% Generate the Model Gradients Function
%
function [gradients,loss] = modelGradients(parameters,dlX,dlT,dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC)
    % Make predictions with the initial conditions.
    U = model(parameters,dlX,dlT);
    % Assume a value for the parameter k
    k = 0.01;
    % Calculate derivatives with respect to X and T.
    gradientsU = dlgradient(sum(U,'all'),{dlX,dlT},'EnableHigherDerivatives',true);
    Ux = gradientsU{1};
    Ut = gradientsU{2};
    % Calculate second-order derivatives with respect to X.
    Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);
    % Calculate second-order derivatives with respect to T.
    Utt = dlgradient(sum(Ut,'all'),dlT,'EnableHigherDerivatives',true);
    % Calculate lossF. Enforce Burger's equation.
    f = Utt -k.*Uxx;
    zeroTarget = zeros(size(f), 'like', f);
    lossF = mse(f, zeroTarget);
    % Calculate lossI. Enforce initial conditions.
    dlUICPred = model(parameters,dlXIC,dlTIC);
    lossI = mse(dlUICPred, dlUIC);
    % Calculate lossV. Loss in the derivative of the initial conditions.
    dlVICPred = model(parameters,dlXIC,dlTIC);
    Un = dlgradient(sum(dlVICPred,'all'),dlTIC,'EnableHigherDerivatives',true);
    lossV = mse(Un,0*Un);
    % Calculate lossB. Enforce boundary conditions. Using Dirichlet BCs
    dlUBCPred = model(parameters,dlXBC,dlTBC);
    Un = dlgradient(sum(dlUBCPred,'all'),dlXBC,'EnableHigherDerivatives',true);
    lossB = mse(dlUBCPred, dlUBC);
    % Combine losses.
    loss = lossF + lossI + lossV + lossB;
    % Calculate gradients with respect to the learnable parameters.
    gradients = dlgradient(loss,parameters);
end
%
% Generate the Model for the Full Network
%
function dlU = model(parameters,dlX,dlT)
    dlXT = [dlX;dlT];
    numLayers = numel(fieldnames(parameters))/2;
    % First fully connect operation.
    weights = parameters.fc1_Weights;
    bias = parameters.fc1_Bias;
    dlU = fullyconnect(dlXT,weights,bias);
    % tanh and fully connect operations for remaining layers.
    for i=2:numLayers
        name = "fc" + i;
        dlU = tanh(dlU);
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
        i = 0;
        className = 'single'
    end
    s = rng(i);
    parameter = sqrt(2/numIn) * randn(sz,className);
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
% Alternate Finite Differences Routine
%
% function u = findiff(x,t,alpha) 
%     % Problem Initialization
%     % Describing the initial condition and boundary conditions
%     phi0 = exp(-alpha*x.^2);
%     phivar = phi0;
%     dt = diff(t); dx = diff(x);
%     % Applying the resolution of the problem
%     beta = 0.1;
%     sz = length(x);
%     r = beta*dt(1)/(dx(1)^2); % for stability, must be 0.5 or less
%     for j = 2:length(t) % for time steps
%         phi = phi0(); 
%         M = diag(-2*ones(1,sz)) + diag(ones(1,sz-1),1) + diag(ones(1,sz-1),-1);
%         phi = phi + r*M\phi';
%         phi0 = phi;
%         phivar = [phivar;phi0];
%     end
%     u = phivar;
% end
% This is the commented original version of the finite differences
% algorithm, which did not account for a more efficient matrix
% implementation.
function u = findiff(x,t,alpha) 
    % Problem Initialization
    % Describing the initial condition and boundary conditions
    phi0 = exp(-alpha*x.^2);
    phiL = 0;
    % phiR = 0;
    phivar = zeros(length(t),length(x));
    phivar(1,:) = phi0;
    dt = diff(t); dx = diff(x);
    % Applying the resolution of the problem
    beta = 0.1;
    r = beta*dt(1)/(dx(1)^2); % for stability, must be 0.5 or less
    for j = 2:length(t) % for time steps
        phi = phi0();        
        for i = 1:length(x) % for space steps
            if i == 1 || i == length(x)
                phi(i) = phiL;
            else
                phi(i) = phi(i)+r*(phi(i+1)-2*phi(i)+phi(i-1));
            end
        end
        phi0 = phi;
        phivar(j,:) = [phivar;phi0];
    end
    u = phivar;
end
%
% Function to generate the parameters struct of the function
%
function [parameters,seed] = generateParameters(numLayers,numNeurons,seed)
% Let us generate a structure for the parameters of the operation so that
% we might describe a variable that depends on the manually inputted
% neurons and layers.
    parameters = struct;
% a.   Generating the parameters for the first operation.
    % The inputs are the initial values of x and t (2-dimensions) for each 
    % of the number of total neurons, therefore numNeurons
    sz = [numNeurons 2];
    parameters.fc1_Weights = initializeHe(sz,2,seed);
    seed = seed + 1;
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
        parameters.(name + "_Weights") = initializeHe(sz,numIn,seed);
        seed = seed + 1;
        parameters.(name + "_Bias") = initializeZeros([numNeurons 1]);
    end
% c.   Generating the parameters for the last fully connect operation.
    sz = [1 numNeurons];
    % The inputs for the last connect operation are also the number of neurons
    numIn = numNeurons; 
    % Initializing the parameters for the connect operation
    parameters.("fc" + numLayers + "_Weights") = initializeHe(sz,numIn,seed);
    seed = seed + 1;
    parameters.("fc" + numLayers + "_Bias") = initializeZeros([1 1]);
end

function [parameters,output,exitflag] = PINN_HE_Dirichlet(in,parameters,options)
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
    dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,parameterNames,parameterSizes);
% Run the fmincon algorithm and obtain the total runtime of the algorithm
tic
%[x,fval,exitflag,output,lambda,grad,hessian]
[parametersV,fval,exitflag,output,lambda,grad,hessian] = ...
    fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
toc
% Revert the optimized version of the parameters to a struct. 
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
end
