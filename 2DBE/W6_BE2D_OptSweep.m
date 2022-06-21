%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   W7 Version of the Code                         %    by Luis Mora Ortiz 
%   Bachelor Thesis                                %    2021/2022          
%   A Deep Learning Approach to Solve Partial Differential Equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   The Partial Differential Equation is the HEAT EQUATION
%   The Activation function is the ARCTANGENT
%   The Boundary Conditions are DIRICHLET
%   The problem is TWO-DIMENSIONAL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This is the code for the swept solution of the Burgers equation in
% two-dimensions. The file with the data to obtain the results of this
% problem is the "Data_BE2D_Sweep.mat".

%% Part I.- Starting the Program and General Definitions
% Could be commented due to the large runtime of the algorithm
clear; clc; close all;

%% a.   Describe the dimensions of the problem
% Set the dimensions in the x-direction
xmin =  0;
xmax =  1;
% Set the dimensions in the y-direction
ymin =  0;
ymax =  1;
% Set the maximum time dimension
tmax =  1;

%% b.   Description of the Boundary Conditions (constraints in space)
% Prescribe the number of boundary conditions at either side (temporal)
tnum = 11; xnum = 21;
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
v0BC1x = [];
v0BC2x = [];

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
u0BC1y = [];
u0BC2y = [];
v0BC1y = zeros(tnum,xnum);
v0BC2y = zeros(tnum,xnum);

%% c.   Description of the Initial Conditions (constraints in time)
% Prescribe the number of initial condition points (spatial)
xnum = 21; ynum = 21;
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
u0IC = sin(pi*x0IC).*cos(pi*y0IC);
v0IC = cos(pi*x0IC).*sin(pi*y0IC);

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
VBCy = [v0BC1y v0BC2y];
% Let us group the Initial Conditions
XICi = x0IC;
YICi = y0IC;
TICi = t0IC;
UICi = u0IC;
VICi = v0IC;

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
% The structure will be such that it requires 2 input channels (x and t)
% and delivers a double output channel u(x,t) v(x,t).

% Let us simplify the components to be passed across the inputs section
% First, the Boundary Conditions
XBCXi = XBCx; 
XBCYi = XBCy;
YBCXi = YBCx; 
YBCYi = YBCy;
TBCXi = TBCx;
TBCYi = TBCy;
UBCi  = UBCx;
VBCi  = VBCy;
XBCX = zeros(1,height(XBCXi)*width(XBCXi));
YBCX = zeros(1,height(XBCXi)*width(XBCXi)); 
TBCX = zeros(1,height(XBCXi)*width(XBCXi));
XBCY = zeros(1,height(XBCXi)*width(XBCXi));
YBCY = zeros(1,height(XBCXi)*width(XBCXi)); 
TBCY = zeros(1,height(XBCXi)*width(XBCXi));
UBC  = zeros(1,height(XBCXi)*width(XBCXi));
VBC  = zeros(1,height(XBCXi)*width(XBCXi));
for i = 1:height(UBCi)
    for j = 1:width(UBCi)
        ii = (i-1)*width(UBCi)+j;
        XBCX(ii) = XBCXi(i,j);
        XBCY(ii) = XBCYi(i,j);
        YBCX(ii) = YBCXi(i,j);
        YBCY(ii) = YBCYi(i,j);
        TBCX(ii) = TBCXi(i,j);
        TBCY(ii) = TBCYi(i,j);
        UBC(ii) = UBCi(i,j);
        VBC(ii) = VBCi(i,j);
    end
end
% Then, the Initial Conditions
XIC = zeros(1,height(XICi)*width(XICi));
YIC = zeros(1,height(XICi)*width(XICi)); 
TIC = zeros(1,height(XICi)*width(XICi));
UIC = zeros(1,height(XICi)*width(XICi));
VIC = zeros(1,height(XICi)*width(XICi));
for i = 1:height(XICi)
    for j = 1:width(XICi)
        ii = (i-1)*width(XICi)+j;
        XIC(ii) = XICi(i,j);
        YIC(ii) = YICi(i,j);
        TIC(ii) = TICi(i,j);
        UIC(ii) = UICi(i,j);
        VIC(ii) = VICi(i,j);
    end
end

%% g. Plotting the Boundary Conditions, Initial Conditions, and Inside Pts

figure
plot3(xpts,ypts,tpts,'g*')
hold on
plot3(XIC,YIC,TIC,'b*')
plot3(XBCX,YBCX,TBCX,'r*')
plot3(XBCY,YBCY,TBCY,'c*')
hold off
xlabel('x-coordinate','Interpreter','latex')
ylabel('y-coordinate','Interpreter','latex')
zlabel('t-coordinate','Interpreter','latex')
grid minor
legend('Collocation Points','Initial Condition','Boundary Conditions U','Boundary Conditions V',...
    'Location','best','Interpreter','latex')

%% Inserting all of these variables into a structure to be passed around
inputs = struct;
% Boundary Conditions (BC)
inputs.XBCX = XBCX;
inputs.YBCX = YBCX;
inputs.TBCX = TBCX;
inputs.XBCY = XBCY;
inputs.YBCY = YBCY;
inputs.TBCY = TBCY;
inputs.UBC = UBC;
inputs.VBC = VBC;
% Initial Conditions (IC)
inputs.XIC = XIC;
inputs.YIC = YIC;
inputs.TIC = TIC;
inputs.UIC = UIC;
inputs.VIC = VIC;
% Internal Point Grid
inputs.dataX = xpts';
inputs.dataY = ypts';
inputs.dataT = tpts';

% g.   Select the options for the fmincon algorithm
options = optimoptions('fmincon',...        % Using the multivariate restricted optimization function fmincon
    'HessianApproximation', 'lbfgs', ...    % Using the Lim-Memory BFGS algorithm for the Hessian
    'MaxIterations', 1000, ...              % Needs to be sufficiently large so as to ensure convergence
    'MaxFunctionEvaluations', 1000, ...     % Needs to be sufficiently large so as to ensure convergence
    'OptimalityTolerance', 1e-3, ...        % By default considering a tolerance of 0.00001
    'Display','iter-detailed',...
    'SpecifyObjectiveGradient', true);      % User-defined gradient for the algorithm); 

%% f. Sweeping the Dimensions of the Neural Network Architecture
% Let us consider a multilayer perceptron architecture which has L fully
% connected operations and N hidden neurons.
% Initializing the matrices for the errors in BCs and ICs
errBC = struct; errIC = struct;
% Describing the matrices for the initial and boundary conditions
numPredictions = 1E2;
xL = linspace(0,1,numPredictions);
yL = linspace(0,1,numPredictions);
xInit = zeros(1,numPredictions^2); 
yInit = zeros(1,numPredictions^2);
tInit = 0.00*zeros(1,numPredictions^2);
for i = 1:width(xL)
    for j = 1:width(yL)
        ii = (i-1)*width(yL)+j;
        xInit(ii) = xL(i);
        yInit(ii) = yL(j);
    end
end
tBnds = yInit;
xBndA = 0.00*ones(1,numPredictions^2);
xBndB = 1.00*ones(1,numPredictions^2);
yBndA = xInit;
yBndB = xInit;
xBndC = xInit;
xBndD = xInit;
yBndC = 0.00*ones(1,numPredictions^2);
yBndD = 1.00*ones(1,numPredictions^2);
% Pre-defining the coefficients for spectral analysis-
for i = 0:101
    for j = 0:101
        Re = 10;
        mu = 1/Re;
        lambda = 1/(4*pi*mu);
        % Cmat(i,j) = integral2(@(x,y) exp(2*lambda*cos(pi*x).*cos(pi*y)).*cos((i-1)*pi*x).*cos((j-1)*pi*y),0,1,0,1);        
        if mod(i+j,2)==1
            Cmat(i+1,j+1) = 0;
        else
            Cmat(i+1,j+1) = besseli((i+j)/2,lambda)*besseli((i-j)/2,lambda);
        end
    end
end
uInit = sin(pi*xInit).*cos(pi*yInit);
vInit = cos(pi*xInit).*sin(pi*yInit);
% Defining the variables for iterations, function evaluations, first order
% optimality, and time needed until convergence.
iter = struct; 
feval = struct; 
fstopt = struct; 
ttaken = struct;
solsIU = struct; 
solsIV = struct; 
solsBU1 = struct;
solsBU2 = struct;
solsBV1 = struct;
solsBV2 = struct;
% Defining a struct which will indicate 1 converged and 0 non-converged
convd = struct;
%%
nL = 10;
nN = 3;
for i = 1:100 % The number of local trials
    % The structure will be such that it requires 3 input channels (x, y and t)
    % and delivers a single output channel u(x,t).
    % Thus, the parameters of the Neural Network are initialized
    parameters = generateParameters(nL,nN);
    % e. Implement the Model Training for the Hyperbolic Arctangent AF
    tic;
    [parameters,output] = PINN_HE_Dirichlet(inputs,parameters,options);
    ttk = toc;
    % First, storing the compliance with the Initial Conditions
    % Make predictions.
    dlXTest = dlarray(xInit,'CB');
    dlYTest = dlarray(yInit,'CB');
    dlTTest = dlarray(tInit,'CB');
    dlUPred = model(parameters,dlXTest,dlYTest,dlTTest,1);
    dlVPred = model(parameters,dlXTest,dlYTest,dlTTest,2);
    solsIU.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = dlUPred;
    solsIV.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = dlVPred;
    err_ITU = mse(dlUPred,uInit);
    err_ITV = mse(dlVPred,vInit);
    errIC.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = (err_ITU+err_ITV)/2;
    % Then, storing the compliance with the Boundary Conditions
    err_BND = 0;
    % Make predictions.
    dlXTest = dlarray(xBndA,'CB');
    dlYTest = dlarray(yBndA,'CB');
    dlTTest = dlarray(tBnds,'CB');
    dlVPred = model(parameters,dlXTest,dlYTest,dlTTest,2);
    solsBV1.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = dlVPred;
    err_BND = err_BND + mse(dlVPred,0*dlVPred);
    dlXTest = dlarray(xBndB,'CB');
    dlYTest = dlarray(yBndB,'CB');
    dlTTest = dlarray(tBnds,'CB');
    dlVPred = model(parameters,dlXTest,dlYTest,dlTTest,2);
    solsBV2.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = dlVPred;
    err_BND = err_BND + mse(dlVPred,0*dlVPred);
    dlXTest = dlarray(xBndC,'CB');
    dlYTest = dlarray(yBndC,'CB');
    dlTTest = dlarray(tBnds,'CB');
    dlUPred = model(parameters,dlXTest,dlYTest,dlTTest,1);
    solsBU1.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = dlUPred;
    err_BND = err_BND + mse(dlUPred,0*dlUPred);
    dlXTest = dlarray(xBndD,'CB');
    dlYTest = dlarray(yBndD,'CB');
    dlTTest = dlarray(tBnds,'CB');
    dlUPred = model(parameters,dlXTest,dlYTest,dlTTest,1);
    solsBU2.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = dlUPred;
    err_BND = err_BND + mse(dlUPred,0*dlUPred);
    errBC.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = err_BND/4;
    % Then storing the iterations, number of function evaluations,
    % time required until convergence, first order of optimality
    iter.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = output.iterations; 
    feval.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = output.funcCount; 
    fstopt.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = output.firstorderopt; 
    ttaken.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = ttk;
    % Lastly generating a struct to indicate convergence
    yyn = (errIC.("NL_" + nL + "_NN_" + nN + "_TR_" + i)<0.02)&&(errBC.("NL_" + nL + "_NN_" + nN + "_TR_" + i)<0.02);
    convd.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = yyn;
    disp("NumLayers: " + nL + " NumNeurons: " + nN + " TrialID: " + i + " Converged: " + yyn + " TTaken: " + ttk);
end

%% Checkmark for results: 
% if the .mat data is loaded, the archive might be opened from here and
% only the sections above will be ran.

% %% Ensuring the Converged Structure is less restrictive
% for nL = 2:10 % The number of layers
%     for nN = 1:numNeuronsMax % The number of neurons
%         for i = 1:100 % The number of local trials
%             yyn = (errIC.("NL_" + nL + "_NN_" + nN + "_TR_" + i)<0.01)&&(errBC.("NL_" + nL + "_NN_" + nN + "_TR_" + i)<0.01);
%             convd.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = yyn;
%         end
%     end
% end
%% Lines Used to Correct an Error in the Original Database
% for i = 1:100 % The number of local trials
%     % The structure will be such that it requires 3 input channels (x, y and t)
%     % and delivers a single output channel u(x,t).
%     % Thus, the parameters of the Neural Network are initialized
%     %parameters = generateParameters(nL,nN);
%     % e. Implement the Model Training for the Hyperbolic Arctangent AF
%     %tic;
%     %[parameters,output] = PINN_HE_Dirichlet(inputs,parameters,options);
%     %ttk = toc;
%     % First, storing the compliance with the Initial Conditions
%     % Make predictions.
%     %dlXTest = dlarray(xInit,'CB');
%     %dlYTest = dlarray(yInit,'CB');
%     %dlTTest = dlarray(tInit,'CB');
%     %dlUPred = model(parameters,dlXTest,dlYTest,dlTTest,1);
%     %dlVPred = model(parameters,dlXTest,dlYTest,dlTTest,2);
%     %solsIU.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = dlUPred;
%     %solsIV.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = dlVPred;
%     %err_ITU = mse(dlUPred,uInit);
%     %err_ITV = mse(dlVPred,vInit);
%     %errIC.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = (err_ITU+err_ITV)/2;
%     % Then, storing the compliance with the Boundary Conditions
%     err_BND = 0;
%     % Make predictions.
%     %dlXTest = dlarray(xBndA,'CB');
%     %dlYTest = dlarray(yBndA,'CB');
%     %dlTTest = dlarray(tBnds,'CB');
%     %dlVPred = model(parameters,dlXTest,dlYTest,dlTTest,2);
%     %solsBV1.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = dlVPred;
%     err_BND = err_BND + mse(solsBV1.("NL_" + nL + "_NN_" + nN + "_TR_" + i),0*dlVPred);
%     %dlXTest = dlarray(xBndB,'CB');
%     %dlYTest = dlarray(yBndB,'CB');
%     %dlTTest = dlarray(tBnds,'CB');
%     %dlVPred = model(parameters,dlXTest,dlYTest,dlTTest,2);
%     %solsBV2.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = dlVPred;
%     err_BND = err_BND + mse(solsBV2.("NL_" + nL + "_NN_" + nN + "_TR_" + i),0*dlVPred);
%     %dlXTest = dlarray(xBndC,'CB');
%     %dlYTest = dlarray(yBndC,'CB');
%     %dlTTest = dlarray(tBnds,'CB');
%     %dlUPred = model(parameters,dlXTest,dlYTest,dlTTest,1);
%     %solsBU1.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = dlUPred;
%     err_BND = err_BND + mse(solsBU1.("NL_" + nL + "_NN_" + nN + "_TR_" + i),0*dlUPred);
%     %dlXTest = dlarray(xBndD,'CB');
%     %dlYTest = dlarray(yBndD,'CB');
%     %dlTTest = dlarray(tBnds,'CB');
%     %dlUPred = model(parameters,dlXTest,dlYTest,dlTTest,1);
%     %solsBU2.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = dlUPred;
%     err_BND = err_BND + mse(solsBU2.("NL_" + nL + "_NN_" + nN + "_TR_" + i),0*dlUPred);
%     errBC.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = err_BND/4;
%     % Then storing the iterations, number of function evaluations,
%     % time required until convergence, first order of optimality
%     %iter.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = output.iterations; 
%     %feval.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = output.funcCount; 
%     %fstopt.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = output.firstorderopt; 
%     %ttaken.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = ttk;
%     % Lastly generating a struct to indicate convergence
%     yyn = (errIC.("NL_" + nL + "_NN_" + nN + "_TR_" + i)<0.05)&&(errBC.("NL_" + nL + "_NN_" + nN + "_TR_" + i)<0.1);
%     convd.("NL_" + nL + "_NN_" + nN + "_TR_" + i) = yyn;
%     disp("NumLayers: " + nL + " NumNeurons: " + nN + " TrialID: " + i + " Converged: " + yyn + " TTaken: " + ttk);
% end
% filename = "Data_17_June_3DBeq";
% save(filename)

%% Generating the average values in a new struct:
        errBCmean = 0; errBCs = [];
        errICmean = 0; errICs = [];
        itermean = 0; iters = [];
        fevalmean = 0; fevals = [];
        fstoptmean = 0; fstopts = [];
        ttakenmean = 0; ttakens = [];
        convdpercent = 0; convds = [];
        for i = 1:100
            errBCmean = errBCmean + errBC.("NL_" + nL + "_NN_" + nN + "_TR_" + i);
            errBCs(i) = errBC.("NL_" + nL + "_NN_" + nN + "_TR_" + i);
            errICmean = errICmean + errIC.("NL_" + nL + "_NN_" + nN + "_TR_" + i);
            errICs(i) = errIC.("NL_" + nL + "_NN_" + nN + "_TR_" + i);
            itermean = itermean + iter.("NL_" + nL + "_NN_" + nN + "_TR_" + i);
            iters(i) = iter.("NL_" + nL + "_NN_" + nN + "_TR_" + i);
            fevalmean = fevalmean + feval.("NL_" + nL + "_NN_" + nN + "_TR_" + i);
            fevals(i) = feval.("NL_" + nL + "_NN_" + nN + "_TR_" + i);
            fstoptmean = fstoptmean + fstopt.("NL_" + nL + "_NN_" + nN + "_TR_" + i);
            fstopts(i) = fstopt.("NL_" + nL + "_NN_" + nN + "_TR_" + i);
            ttakenmean = ttakenmean + ttaken.("NL_" + nL + "_NN_" + nN + "_TR_" + i);
            ttakens(i) = ttaken.("NL_" + nL + "_NN_" + nN + "_TR_" + i);
            convdpercent = convdpercent + convd.("NL_" + nL + "_NN_" + nN + "_TR_" + i);
            convds(i) = convd.("NL_" + nL + "_NN_" + nN + "_TR_" + i);
        end
        errBCmean = errBCmean/100;
        errICmean = errICmean/100;
        itermean = itermean/100;
        fevalmean = fevalmean/100;
        fstoptmean = fstoptmean/100;
        ttakenmean = ttakenmean/100;
        %convdpercent = convdpercent;

%% Scatter Plot of Initial and Boundary Conditions
figure
hold on
scatter(errICs(convds==1),errBCs(convds==1),'bo');
scatter(errICs((convds==0)&(errBCs>0.001)),errBCs((convds==0)&(errBCs>0.001)),'gs');
scatter(errICs((convds==0)&(errBCs<0.001)),errBCs((convds==0)&(errBCs<0.001)),'rx');
hold off
xlabel('Error in the Initial Conditions','Interpreter','latex');
ylabel('Error in the Boundary Conditions','Interpreter','latex');
set(gca,'YScale','log'); set(gca,'XScale','log');
%err = sum(convds==1); str = "Efficiency of the Algorithm: " + err + "/100";
%title(str,'Interpreter','latex')
legend('Converged','Not Enough Iterations','Not Converged','Location','best','Interpreter','latex')
grid minor
box on
xlim([5e-3,2e-1])

figure
hold on
scatter(iters(convds==1),ttakens(convds==1),'bo');
scatter(iters((convds==0)&(errBCs>0.001)),ttakens((convds==0)&(errBCs>0.001)),'gs');
scatter(iters((convds==0)&(errBCs<0.001)),ttakens((convds==0)&(errBCs<0.001)),'rx');
hold off
xlabel('Number of Iterations','Interpreter','latex');
ylabel('Time Elapsed','Interpreter','latex');
set(gca,'YScale','log'); set(gca,'XScale','log');
%err = sum(convds==1); str = "Efficiency of the Algorithm: " + err + "/100";
%title(str,'Interpreter','latex')
legend('Converged','Not Enough Iterations','Not Converged','Location','best','Interpreter','latex')
grid minor
box on

% %%
% figure
% uex = [];
% for i = 1:100
%     SS = solsBU1.("NL_10_NN_3_TR_" + i);
%     for k = 1:length(SS)
%         jj = mod(k,width(xL));
%         if jj == 0
%             jj = width(xL);
%         end
%         ii = floor(k/width(xL))+1;
%         uex(ii,jj) = SS(k);
%     end
%     subplot(10,10,i);
%     contourf(uex);
%     disp(i);
% end

%% Extra Functions
%
% Minimize the Reynolds number
%
function Xmin = minimizeMesh(Xx,Yy,Uu,Vv,Cmat,Re)
    ff = (Uu - burgers_analU(Xx,Yy,1,Cmat,Re)).^2;
    gg = (Vv - burgers_analV(Xx,Yy,1,Cmat,Re)).^2;
    uu = sum(ff,'all');
    vv = sum(gg,'all');
    Xmin = uu + vv;
end
%
% Analytical Solution of the Problem
%
function [Uu,Vv] = burgers_anal(dlX,dlY,dlT,Cmn)
    Re = 10;
    mu = 1/Re;
    lambda = 1/(4*pi*mu); 
    psi = 0; ut = 0; vt = 0;
    for i = 0:(height(Cmn)-1)
        for j = 0:(width(Cmn)-1)
            Amn = (2-(i==0))*(2-(j==0));
            Emn = exp(-(i^2+j^2)*pi^2*dlT*mu);
            psi = psi + Amn.*Cmn(i+1,j+1).*Emn.*cos(i*pi*dlX).*cos(j*pi*dlY);
            ut = ut + i*Amn.*Cmn(i+1,j+1).*Emn.*sin(i*pi*dlX).*cos(j*pi*dlY);
            vt = vt + j*Amn.*Cmn(i+1,j+1).*Emn.*cos(i*pi*dlX).*sin(j*pi*dlY);
        end
    end
    Uu = (2*pi*mu)*ut./psi;
    Vv = (2*pi*mu)*vt./psi;
end
function Uu = burgers_analU(dlX,dlY,dlT,Cmn,Re)
    % Re = 100;
    mu = 1/Re;
    lambda = 1/(4*pi*mu); 
    psi = 0; ut = 0; vt = 0;
    for i = 0:(height(Cmn)-1)
        for j = 0:(width(Cmn)-1)
            Amn = (2-(i==0))*(2-(j==0));
            Emn = exp(-(i^2+j^2)*pi^2*dlT*mu);
            psi = psi + Amn.*Cmn(i+1,j+1).*Emn.*cos(i*pi*dlX).*cos(j*pi*dlY);
            ut = ut + i*Amn.*Cmn(i+1,j+1).*Emn.*sin(i*pi*dlX).*cos(j*pi*dlY);
        end
    end
    Uu = (2*pi*mu)*ut./psi;
end
function Vv = burgers_analV(dlX,dlY,dlT,Cmn,Re)
    % Re = 100;
    mu = 1/Re;
    lambda = 1/(4*pi*mu); 
    psi = 0; ut = 0; vt = 0;
    for i = 0:(height(Cmn)-1)
        for j = 0:(width(Cmn)-1)
            Amn = (2-(i==0))*(2-(j==0));
            Emn = exp(-(i^2+j^2)*pi^2*dlT*mu);
            psi = psi + Amn.*Cmn(i+1,j+1).*Emn.*cos(i*pi*dlX).*cos(j*pi*dlY);
            vt = vt + j*Amn.*Cmn(i+1,j+1).*Emn.*cos(i*pi*dlX).*sin(j*pi*dlY);
        end
    end
    Vv = (2*pi*mu)*vt./psi;
end

%
% fmincon Objective Function
%
function [loss,gradientsV] = objectiveFunction(parametersU,dlX,dlY,dlT,dlXIC,dlYIC,dlTIC,dlUIC,dlVIC,dlXBCu,dlYBCu,dlTBCu,dlXBCv,dlYBCv,dlTBCv,dlUBC,dlVBC,parameterNames,parameterSizes)
    % Rate of Decay
    a = 0.9;
    % Convert parameters to structure of dlarray objects.
    parametersU = dlarray(parametersU);
    parametersu = parameterVectorToStruct(parametersU,parameterNames,parameterSizes);
    % Evaluate model gradients and loss.
    [gradients,loss] = dlfeval(@modelGradients,parametersu,dlX,dlY,dlT,dlXIC,dlYIC,dlTIC,dlUIC,dlVIC,dlXBCu,dlYBCu,dlTBCu,dlXBCv,dlYBCv,dlTBCv,dlUBC,dlVBC);
    % Return loss and gradients for fmincon.
    gradientsV = parameterStructToVector(gradients);
    gradientsV = a*extractdata(gradientsV);
    loss = extractdata(loss);
end
%
% Generate the Model Gradients Function
%
function [gradients,loss] = modelGradients(parameters,dlX,dlY,dlT,dlXIC,dlYIC,dlTIC,dlUIC,dlVIC,dlXBCu,dlYBCu,dlTBCu,dlXBCv,dlYBCv,dlTBCv,dlUBC,dlVBC)
    % Make predictions with the initial conditions.
    UU = model(parameters,dlX,dlY,dlT,1);
    VV = model(parameters,dlX,dlY,dlT,2);
    % Assume a value for the parameter Re
    Re = 10;
    % Calculate derivatives with respect to X and T.
    gradientsU = dlgradient(sum(UU,'all'),{dlX,dlY,dlT},'EnableHigherDerivatives',true); % +dlXBC+dlXIC//+dlTIC+dlTBC
    Ux = gradientsU{1};
    Uy = gradientsU{2};
    Ut = gradientsU{3};
    % Calculate second-order derivatives with respect to X.
    Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true); % +dlXBC+dlXIC
    Uyy = dlgradient(sum(Uy,'all'),dlY,'EnableHigherDerivatives',true); % +dlXBC+dlXIC
    % Calculate second-order derivatives with respect to T
    Utt = dlgradient(sum(Ut,'all'),dlT,'EnableHigherDerivatives',true);
    % Calculate derivatives with respect to X and T.
    gradientsV = dlgradient(sum(VV,'all'),{dlX,dlY,dlT},'EnableHigherDerivatives',true); % +dlXBC+dlXIC//+dlTIC+dlTBC
    Vx = gradientsV{1};
    Vy = gradientsV{2};
    Vt = gradientsV{3};
    % Calculate second-order derivatives with respect to X.
    Vxx = dlgradient(sum(Vx,'all'),dlX,'EnableHigherDerivatives',true); % +dlXBC+dlXIC
    Vyy = dlgradient(sum(Vy,'all'),dlY,'EnableHigherDerivatives',true); % +dlXBC+dlXIC
    % Calculate second-order derivatives with respect to T
    Vtt = dlgradient(sum(Vt,'all'),dlT,'EnableHigherDerivatives',true);
    % Calculate lossF. Enforce Burger's equation.
    f = Ut + UU.*Ux + VV.*Uy - 1/Re*(Uxx + Uyy);
    zeroTarget = zeros(size(f), 'like', f);
    lossF = mse(f, zeroTarget);
    % Calculate lossG. Enforce Burger's equation.
    g = Vt + UU.*Vx + VV.*Vy - 1/Re*(Vxx + Vyy);
    zeroTarget = zeros(size(g), 'like', g);
    lossG = mse(g, zeroTarget);
    % Calculate lossI. Enforce initial conditions.
    dlUICPred = model(parameters,dlXIC,dlYIC,dlTIC,1);
    lossI = mse(dlUICPred, dlUIC);
    dlVICPred = model(parameters,dlXIC,dlYIC,dlTIC,2);
    lossJ = mse(dlVICPred, dlVIC);
    % Calculate lossB. Enforce boundary conditions. Using Dirichlet BCs
    dlUBCPred = model(parameters,dlXBCu,dlYBCu,dlTBCu,1);
    Un = dlgradient(sum(dlUBCPred,'all'),dlYBCu,'EnableHigherDerivatives',true);
    lossB = mse(dlUBCPred, dlUBC);
    % Calculate lossW. Enforce boundary conditions. Using Dirichlet BCs
    dlVBCPred = model(parameters,dlXBCv,dlYBCv,dlTBCv,2);
    Vn = dlgradient(sum(dlVBCPred,'all'),dlXBCv,'EnableHigherDerivatives',true);
    lossW = mse(dlVBCPred, dlVBC);
    % Combine losses.
    loss = (lossF + lossG) + (lossI + lossJ) + (lossB + lossW);
    % Calculate gradients with respect to the learnable parameters.
    gradients = dlgradient(loss,parameters);
end
%
% Generate the Model for the Full Network
%
function dlU = model(parameters,dlX,dlY,dlT,a)
    dlXT = [dlX;dlY;dlT];
    numLayers = numel(fieldnames(parameters))/4;
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
    sz = [numNeurons 3];
    parameters.fc1_1_Weights = initializeHe(sz,3);
    parameters.fc1_1_Bias = initializeZeros([numNeurons 1]);
    parameters.fc1_2_Weights = initializeHe(sz,3);
    parameters.fc1_2_Bias = initializeZeros([numNeurons 1]);
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
end

function [parameters,output] = PINN_HE_Dirichlet(in,parameters,options)
% Part III.- Train the Network Using fmincon
% Generate the vector with the parameters to be inserted in the algorithm.
[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters); 
parametersV = double(extractdata(parametersV));

% Generate the arrays with the data for loss function minimization

% Boundary Conditions (BC)
dlXBCu = dlarray(in.XBCX,'CB');
dlYBCu = dlarray(in.YBCX,'CB');
dlTBCu = dlarray(in.TBCX,'CB');
dlXBCv = dlarray(in.XBCY,'CB');
dlYBCv = dlarray(in.YBCY,'CB');
dlTBCv = dlarray(in.TBCY,'CB');
dlUBC = dlarray(in.UBC,'CB');
dlVBC = dlarray(in.VBC,'CB');
% Initial Conditions (IC)
dlXIC = dlarray(in.XIC,'CB');
dlYIC = dlarray(in.YIC,'CB');
dlTIC = dlarray(in.TIC,'CB');
dlUIC = dlarray(in.UIC,'CB');
dlVIC = dlarray(in.VIC,'CB');
% Internal Point Grid
dlX = dlarray(in.dataX','CB');
dlY = dlarray(in.dataY','CB');
dlT = dlarray(in.dataT','CB');

% Define the objective function through the data obtained beforehand
objFun = @(parameters) objectiveFunction(parameters,dlX,dlY,dlT,...
    dlXIC,dlYIC,dlTIC,dlUIC,dlVIC,dlXBCu,dlYBCu,dlTBCu,...
    dlXBCv,dlYBCv,dlTBCv,dlUBC,dlVBC,parameterNames,parameterSizes);

% Run the fmincon algorithm and obtain the total runtime of the algorithm
tic
%[x,fval,exitflag,output,lambda,grad,hessian]
[parametersV,~,~,output,~,~,~] = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
toc
% Revert the optimized version of the parameters to a struct. 
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
end
