%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   W7 Version of the Code                         %    by Luis Mora Ortiz 
%   Bachelor Thesis                                %    2021/2022          
%   A Deep Learning Approach to Solve Partial Differential Equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   The Partial Differential Equation is the BURGERS EQUATION
%   The Activation function is the ARCTANGENT
%   The Boundary Conditions are DIRICHLET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This is the baseline code for the 2DBE.

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
u0IC = 1.1*sin(pi*x0IC).*cos(pi*y0IC);
v0IC = 1.1*cos(pi*x0IC).*sin(pi*y0IC);

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
% Let us consider a multilayer perceptron architecture which has L fully
% connected operations and N hidden neurons.
numLayers = 20;
numNeurons = 10;

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

% Thus, the parameters of the Neural Network are initialized
parameters = generateParameters(numLayers,numNeurons);
% Redefining the initial parameters in a new structure for it to be kept.
parameters_init = parameters;

%% g.   Select the options for the fmincon algorithm
options = optimoptions('fmincon', ... % Using the multivariate restricted optimization function fmincon
    'HessianApproximation','lbfgs', ... % Using the Lim-Memory BFGS algorithm for the Hessian
    'MaxIterations',1e3, ... % Needs to be sufficiently large so as to ensure convergence
    'MaxFunctionEvaluations',2e3, ... % Needs to be sufficiently large so as to ensure convergence
    'OptimalityTolerance',1e-3, ... % By default considering a tolerance of 0.00001
    'SpecifyObjectiveGradient',true,... % User-defined gradient for the algorithm
    'PlotFcn',@optimplotfirstorderopt,... % Plotting the first-order optimality across iterations
    'Display','iter-detailed'); % Deliver the number of needed iterations

% e. Implement the Model Training for the Hyperbolic Arctangent AF
[parameters_fin,output] = PINN_HE_Dirichlet(inputs,parameters,options);

%% Plotting only the initial conditions
numPredictions = 1E2+1;
XT = linspace(0,1,numPredictions);
YT = linspace(0,1,numPredictions);
XTest = zeros(1,numPredictions^2); 
YTest = zeros(1,numPredictions^2);
for i = 1:width(XT)
    for j = 1:width(YT)
        ii = (i-1)*width(YT)+j;
        XTest(ii) = XT(i);
        YTest(ii) = YT(j);
    end
end
TTest = 0*ones(size(XTest));

dlXTest = dlarray(XTest,'CB');
dlYTest = dlarray(YTest,'CB');
dlTTest = dlarray(TTest,'CB');
dlUPred = model(parameters_fin,dlXTest,dlYTest,dlTTest,1);
dlVPred = model(parameters_fin,dlXTest,dlYTest,dlTTest,2);

Xx = [];
Yy = [];
Uu = [];
Vv = [];
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
end

figure
clf
subplot(3,2,1)
surf(Xx,Yy,sin(pi*Xx).*cos(pi*Yy))
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('u','Interpreter','latex')
title('Initial Condition $u(x,y) = sin(\pi\cdot x)\cdot cos(\pi\cdot y)$','Interpreter','latex')
colorbar;
subplot(3,2,2)
surf(Xx,Yy,cos(pi*Xx).*sin(pi*Yy))
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('v','Interpreter','latex')
title('Initial Condition $v(x,y) = cos(\pi\cdot x)\cdot sin(\pi\cdot y)$','Interpreter','latex')
colorbar;
subplot(3,2,3)
surf(Xx,Yy,Uu)
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('u','Interpreter','latex')
title('Obtained Velocity u','Interpreter','latex')
colorbar;
subplot(3,2,4)
surf(Xx,Yy,Vv)
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('v','Interpreter','latex')
title('Obtained Velocity v','Interpreter','latex')
colorbar;
error_U_IC = mse(Uu-sin(pi*Xx).*cos(pi*Yy))
error_V_IC = mse(Vv-cos(pi*Xx).*sin(pi*Yy))
subplot(3,2,5)
surf(Xx,Yy,Uu-sin(pi*Xx).*cos(pi*Yy))
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('u','Interpreter','latex')
str = "Difference in Velocities (u) with error = " + error_U_IC; 
title(str,'Interpreter','latex')
colorbar;
subplot(3,2,6)
surf(Xx,Yy,Vv-cos(pi*Xx).*sin(pi*Yy))
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('v','Interpreter','latex')
str = "Difference in Velocities (v) with error = " + error_V_IC; 
title(str,'Interpreter','latex')
colorbar;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Same Plot as above but formatted for the TFG
figure
clf
subplot(2,3,1)
surf(Xx,Yy,sin(pi*Xx).*cos(pi*Yy))
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('u','Interpreter','latex')
title('Initial Condition $u(x,y) = sin(\pi\cdot x)\cdot cos(\pi\cdot y)$','Interpreter','latex')
colorbar;
caxis([-1,1]);
shading interp
subplot(2,3,4)
surf(Xx,Yy,cos(pi*Xx).*sin(pi*Yy))
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('v','Interpreter','latex')
title('Initial Condition $v(x,y) = cos(\pi\cdot x)\cdot sin(\pi\cdot y)$','Interpreter','latex')
colorbar;
caxis([-1,1]);
shading interp
subplot(2,3,2)
surf(Xx,Yy,Uu)
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('u','Interpreter','latex')
title('Obtained Velocity u','Interpreter','latex')
colorbar;
caxis([-1,1]);
shading interp
subplot(2,3,5)
surf(Xx,Yy,Vv)
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('v','Interpreter','latex')
title('Obtained Velocity v','Interpreter','latex')
colorbar;
caxis([-1,1]);
shading interp
error_U_IC = mse(Uu-sin(pi*Xx).*cos(pi*Yy))
error_V_IC = mse(Vv-cos(pi*Xx).*sin(pi*Yy))
subplot(2,3,3)
surf(Xx,Yy,abs(Uu-sin(pi*Xx).*cos(pi*Yy)))
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('u','Interpreter','latex')
str = "Difference in Velocities (u) with error = " + error_U_IC; 
title(str,'Interpreter','latex')
colorbar;
shading interp
caxis([0,max(max(abs(Uu-sin(pi*Xx).*cos(pi*Yy))'))]);
subplot(2,3,6)
surf(Xx,Yy,abs(Vv-cos(pi*Xx).*sin(pi*Yy)))
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('v','Interpreter','latex')
str = "Difference in Velocities (v) with error = " + error_V_IC; 
title(str,'Interpreter','latex')
colorbar;
shading interp
caxis([0,max(max(abs(Vv-cos(pi*Xx).*sin(pi*Yy))'))]);

%% Plotting only the boundary conditions
numPredictions = 1E2+1;
XT = linspace(0,1,numPredictions);
TT = linspace(0,1,numPredictions);
XTest = zeros(1,numPredictions^2); 
TTest = zeros(1,numPredictions^2);
for i = 1:width(XT)
    for j = 1:width(TT)
        ii = (i-1)*width(TT)+j;
        XTest(ii) = XT(i);
        TTest(ii) = TT(j);
    end
end
YTest1 = 0*ones(size(XTest));
dlXTest = dlarray(XTest,'CB');
dlYTest1 = dlarray(YTest1,'CB');
dlTTest = dlarray(TTest,'CB');
Vleft = extractdata(model(parameters_fin,dlXTest,dlYTest1,dlTTest,2));
YTest2 = ones(size(XTest));
dlYTest2 = dlarray(YTest2,'CB');
Vright = extractdata(model(parameters_fin,dlXTest,dlYTest2,dlTTest,2));

numPredictions = 1E2+1;
YT = linspace(0,1,numPredictions);
TT = linspace(0,1,numPredictions);
YTest = zeros(1,numPredictions^2); 
TTest = zeros(1,numPredictions^2);
for i = 1:width(YT)
    for j = 1:width(TT)
        ii = (i-1)*width(TT)+j;
        YTest(ii) = YT(i);
        TTest(ii) = TT(j);
    end
end
XTest1 = 0*ones(size(YTest));
dlYTest = dlarray(YTest,'CB');
dlXTest1 = dlarray(XTest1,'CB');
dlTTest = dlarray(TTest,'CB');
Uleft = extractdata(model(parameters_fin,dlXTest1,dlYTest,dlTTest,1));
XTest2 = ones(size(YTest));
dlXTest2 = dlarray(XTest2,'CB');
Uright = extractdata(model(parameters_fin,dlXTest2,dlYTest,dlTTest,1));

UL = []; VL = []; UR = []; VR = [];
for s = 1:length(dlTTest)
    ii = ceil(s/width(YT)); % iii(s) = ii;
    jj = mod(s,width(YT));  % jjj(s) = jj;
    if ~jj
        jj = width(YT);
    end
    UL(ii,jj) = Uleft(s);
    VL(ii,jj) = Vleft(s);
    UR(ii,jj) = Uright(s);
    VR(ii,jj) = Vright(s);
end

figure
subplot(2,2,1)
surf(XT,TT,abs(VL));
xlabel('x-coordinate','Interpreter','latex');
ylabel('t-coordinate','Interpreter','latex');
zlabel('v-value','Interpreter','latex');
str = "Boundary Condition $V_{y=0}=0$ with error = " + mse(Vleft);
title(str,'Interpreter','latex')
colorbar
caxis([0,0.2]);
shading interp

subplot(2,2,2)
surf(XT,TT,abs(VR));
xlabel('x-coordinate','Interpreter','latex');
ylabel('t-coordinate','Interpreter','latex');
zlabel('v-value','Interpreter','latex');
str = "Boundary Condition $V_{y=1}=0$ with error = " + mse(Vright);
title(str,'Interpreter','latex')
colorbar
caxis([0,0.2]);
shading interp

subplot(2,2,3)
surf(YT,TT,abs(UL));
xlabel('y-coordinate','Interpreter','latex');
ylabel('t-coordinate','Interpreter','latex');
zlabel('u-value','Interpreter','latex');
str = "Boundary Condition $U_{y=0}=0$ with error = " + mse(Uleft);
title(str,'Interpreter','latex')
colorbar
caxis([0,0.2]);
shading interp

subplot(2,2,4)
surf(YT,TT,abs(UR));
xlabel('y-coordinate','Interpreter','latex');
ylabel('t-coordinate','Interpreter','latex');
zlabel('u-value','Interpreter','latex');
str = "Boundary Condition $U_{y=1}=0$ with error = " + mse(Uright);
title(str,'Interpreter','latex')
colorbar
caxis([0,0.2]);
shading interp

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Let us plot some results:
% Let us first determine a set of particular time instances at which to
% study the model.
tTest = [0.00 0.10 0.25 0.50];
numPredictions = 1E2+1;
XT = linspace(0,1,numPredictions);
YT = linspace(0,1,numPredictions);
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
    dlVPred = model(parameters_fin,dlXTest,dlYTest,dlTTest,2);

    Xx = [];
    Yy = [];
    Uu = [];
    Vv = [];
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
    end

    % Plot predictions.
    subplot(4,2,2*i-1)
    surf(Yy,Xx,Uu);
    title("Obtained U for t = " + t,'Interpreter','latex');
    colorbar
    subplot(4,2,2*i)
    surf(Yy,Xx,Vv);
    title("Obtained V for t = " + t,'Interpreter','latex');
    colorbar
end

%%
figure
for i = 0:101
    for j = 0:101
        Re = 100;
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
for i = 1:numel(tTest)
    t = tTest(i);
%     UPred = [];
%     VPred = []; 
%     for s = 1:length(dlXTest)
%         [UPred(s),VPred(s)] = burgers_anal(XTest(s),YTest(s),t,Cmat);
%     end
    [UPred,VPred] = burgers_anal(dlXTest,dlYTest,t,Cmat);
    Uu2 = [];
    Vv2 = [];
    for s = 1:length(dlXTest)
        ii = ceil(s/width(YT)); % iii(s) = ii;
        jj = mod(s,width(YT));  % jjj(s) = jj;
        if ~jj
            jj = width(YT);
        end
        Xx(ii,jj) = XT(ii);
        Yy(ii,jj) = YT(jj);
        Uu2(ii,jj) = UPred(s);
        Vv2(ii,jj) = VPred(s);
    end
    % Plot predictions.
    subplot(4,2,2*i-1)
    Uu2(abs(Uu2)>1) = NaN;
    surf(Yy,Xx,Uu2);
    title("Expected U for t = " + t,'Interpreter','latex');
    zlim([-1,1])
    colorbar
    subplot(4,2,2*i)
    Vv2(abs(Vv2)>1) = NaN;
    surf(Yy,Xx,Vv2);
    title("Expected V for t = " + t,'Interpreter','latex');
    colorbar
    zlim([-1,1])
end

%% Plotting only the final conditions
numPredictions = 1E2+1;
XT = linspace(0,1,numPredictions);
YT = linspace(0,1,numPredictions);
XTest = zeros(1,numPredictions^2); 
YTest = zeros(1,numPredictions^2);
for i = 1:width(XT)
    for j = 1:width(YT)
        ii = (i-1)*width(YT)+j;
        XTest(ii) = XT(i);
        YTest(ii) = YT(j);
    end
end
TTest = 1*ones(size(XTest));

dlXTest = dlarray(XTest,'CB');
dlYTest = dlarray(YTest,'CB');
dlTTest = dlarray(TTest,'CB');
dlUPred = model(parameters_fin,dlXTest,dlYTest,dlTTest,1);
dlVPred = model(parameters_fin,dlXTest,dlYTest,dlTTest,2);

Xx = [];
Yy = [];
Uu = [];
Vv = [];
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
end

figure
clf
[Ux,Vy] = burgers_anal(Xx,Yy,1,Cmat);
subplot(2,3,1)
surf(Xx,Yy,Ux)
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('u','Interpreter','latex')
title('Final Condition $u(x,y)$','Interpreter','latex')
colorbar;
shading interp
caxis([min(min((Ux))'),max(max((Ux))')]);
subplot(2,3,4)
surf(Xx,Yy,Vy)
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('v','Interpreter','latex')
title('Final Condition $v(x,y)$','Interpreter','latex')
colorbar;
shading interp
caxis([min(min((Vy))'),max(max((Vy))')]);
subplot(2,3,2)
surf(Xx,Yy,Uu)
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('u','Interpreter','latex')
title('Obtained Velocity u','Interpreter','latex')
colorbar;
shading interp
caxis([min(min((Ux))'),max(max((Ux))')]);
subplot(2,3,5)
surf(Xx,Yy,Vv)
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('v','Interpreter','latex')
title('Obtained Velocity v','Interpreter','latex')
colorbar;
shading interp
caxis([min(min((Vy))'),max(max((Vy))')]);
error_U_IC = mse(Uu-Ux)
error_V_IC = mse(Vv-Vy)
subplot(2,3,3)
surf(Xx,Yy,Uu-Ux)
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('u','Interpreter','latex')
str = "Difference in Velocities (u) with error = " + error_U_IC; 
title(str,'Interpreter','latex')
colorbar;
shading interp
caxis([0,max(max(abs(Uu-Ux)))]);
subplot(2,3,6)
surf(Xx,Yy,Vv-Vy)
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('v','Interpreter','latex')
str = "Difference in Velocities (v) with error = " + error_V_IC; 
title(str,'Interpreter','latex')
colorbar;
shading interp
caxis([0,max(max(abs(Vv-Vy)))]);

% %% Obtain the effective Reynolds number
% 
% x0 = 30;
% options = optimoptions('fmincon', ... % Using the multivariate restricted optimization function fmincon
%     'HessianApproximation','lbfgs', ... % Using the Lim-Memory BFGS algorithm for the Hessian
%     'MaxIterations',8000, ... % Needs to be sufficiently large so as to ensure convergence
%     'MaxFunctionEvaluations',8000, ... % Needs to be sufficiently large so as to ensure convergence
%     'OptimalityTolerance',1e-3, ... % By default considering a tolerance of 0.00001
%     'SpecifyObjectiveGradient',true,... % User-defined gradient for the algorithm
%     'PlotFcn',@optimplotfirstorderopt,... % Plotting the first-order optimality across iterations
%     'Display','iter-detailed'); % Deliver the number of needed iterations

% X1 = 0:0.01:1; 
% Y1 = 0:0.01:1;
% Xx = [];
% Yy = [];
% 
% for i = 1:length(X1)
%     Yy = [Yy, Y1'];
% end
% for j = 1:length(Y1)
%     Xx = [Xx; X1];
% end
% 
% Res = 10:1:200;
% fmin = Res*0;
% for i = 1:length(Res)
%     fmin(i) = minimizeMesh(Xx,Yy,Uu,Vv,Cmat,Res(i));
%     disp(i)
% end
% 
% %% Represent the minimization dependent on Re
% figure
% %fmin2 = fmin*(101*101*2);
% plot(Res,fmin);
% xlabel('Reynolds Number ($Re$)','Interpreter','latex');
% ylabel('Mean Error per Element','Interpreter','latex');
% [a,b] = min(fmin)
% grid minor
% 
% % Does not converge at the moment.
% % [a,b] = fmincon(@(Re) minimizeMesh(Xx,Yy,Uu,Vv,Cmat,Re),x0,[],[],[],[],[],[],[],options);

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
    Re = 100;
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
    Re = 100;
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
    loss = 0.5*(lossF + lossG) + 1.3*(lossI + lossJ) + 0.7*(lossB + lossW);
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
    parameters.fc1_2_Weights = initializeHe(sz,3,1);
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

function a = logtransf(a)
    a = round(256./(1+exp(-a)));
end