%% ============================================================
% INSURANCE_MATLAB.m
% Weather derivatives on "Acqua-Alta" in Venice
% ============================================================

clear; clc; close all;

%% loading of the different zones

if ~isfile('zones_sestieri.mat')
    error('File zones_sestieri.mat not found. run BUILD_SESTIERI.m');
end

load('zones_sestieri.mat','zones');


%% Parameters of the product
r    = 0.025;   % risk-free rate
T    = 1;    % maturity of the derivative
Nsim = 5e4;    % number of Monte Carlo simulations
derivative_type = menu("Choose the model type", "Proportional model",...
    "Binomial model");

%% Historical data
filename = 'valori_2013_2023.xlsx';
[~, sheets] = xlsfinfo(filename);
Data = [];
for k = 1:numel(sheets)
    Values = readtable(filename, 'Sheet', sheets{k});
    Data = [Data;Values];
end

mu = mean(Data); % mean of the Historical data
sigma = std(Data); % standard deviation of the Historical data
mu = table2array(mu); % convert the data from a table to a matrix
sigma = table2array(sigma);

cost_per_year = 800000; % total cost per year for mobile walkways
years_hist = 11; % number of years of historical data


% Cost per zone
C_SM = cost_per_year * 35/100; % San Marco
C_Cannaregio = cost_per_year * 15/100;
C_Castello = cost_per_year * 15/100;
C_SP = cost_per_year * ((35/3)/100); % San Polo
C_D = cost_per_year * ((35/3)/100); % Dorsoduro
C_SC = cost_per_year * ((35/3)/100); % Santa Croce


%% Computation for zone
for i = 1:numel(zones)
    switch lower(zones(i).name)
        case 'cannaregio'
            zones(i).mu      = mu;
            zones(i).sigma   = sigma;
            zones(i).strike  = 110;
            zones(i).notional = C_Cannaregio;

        case 'san marco'
            zones(i).mu      = mu;
            zones(i).sigma   = sigma;
            zones(i).strike  = 100;
            zones(i).notional = C_SM;

        case 'san polo'
            zones(i).mu      = mu;
            zones(i).sigma   = sigma;
            zones(i).strike  = 120;
            zones(i).notional = C_SP;

        case 'dorsoduro'
            zones(i).mu      = mu;
            zones(i).sigma   = sigma;
            zones(i).strike  = 120;
            zones(i).notional = C_D;

        case 'castello'
            zones(i).mu      = mu;
            zones(i).sigma   = sigma;
            zones(i).strike  = 110;
            zones(i).notional = C_Castello;

        case 'santa croce'
            zones(i).mu      = mu;
            zones(i).sigma   = sigma;
            zones(i).strike  = 120;
            zones(i).notional = C_SC;

        otherwise
            zones(i).mu      = mu;
            zones(i).sigma   = sigma;
            zones(i).strike  = 150;
            zones(i).notional = cost_per_year;
    end
end

%% Computation of the payoff and price of the derivatives
switch derivative_type
    case 1 % Proportional model
        for i = 1:numel(zones)
            [zones(i).price, ~, simPayoff] = priceZoneDerivative...
                (zones(i), Nsim, r, T);
            zones(i).payoff_mean = mean(simPayoff)
        end
        disp('Prices of the proportional model derivative per zone:');
        for i = 1:numel(zones)
            fprintf('%-12s : %.2f €\n', zones(i).name, zones(i).price);
        end

        disp('Payoffs of the proportional model derivative per zone:');
        for i = 1:numel(zones)
            fprintf('%-12s : %.2f €\n', zones(i).name, zones(i).payoff_mean);
        end
    case 2 % Binomial model
        for i = 1:numel(zones)
            [zones(i).price, ~, simPayoff] = priceZoneDerivative_bin...
                (zones(i), Nsim, r, T);
            zones(i).payoff_mean = mean(simPayoff);
        end
        disp('Prices of the binomial model derivative per zone:');
        for i = 1:numel(zones)
            fprintf('%-12s : %.2f €\n', zones(i).name, zones(i).price);
        end

        disp('Payoffs of the binomial model derivative per zone:');
        for i = 1:numel(zones)
            fprintf('%-12s : %.2f €\n', zones(i).name, zones(i).payoff_mean);
        end

end


%% Build the figure

figure('Name','Weather Derivative - Sestieri di Venezia');
hold on; axis equal;
title('Color Map representing different sestieri of Venice');
xlabel(''); ylabel('');
set(gca,'YDir','reverse');  

allP = [zones.price];
minP = min(allP);
maxP = max(allP);
cmap = turbo(256);

switch derivative_type
    case 1
        for i = 1:numel(zones)
            x = zones(i).x;
            y = zones(i).y;

            if maxP == minP
                cNorm = 0.5;
            else
                cNorm = (zones(i).price - minP) / (maxP - minP);
            end
            idxColor = max(1, min(256, round(cNorm*255)+1));

            h = patch(x, y, cmap(idxColor,:), ...
                'EdgeColor','k','FaceAlpha',0.75);

            zones(i).handle = h;
            set(h,'UserData',i);
            set(h,'ButtonDownFcn', @(src,event) zoneClicked(src,event,zones,r,T,Nsim,cost_per_year));

            xc = mean(x);
            yc = mean(y);
            text(xc-10, yc-20, zones(i).name, ...
                'HorizontalAlignment','center', ...
                'FontWeight','bold', 'Color','w');
        end
    case 2
        for i = 1:numel(zones)
            x = zones(i).x;
            y = zones(i).y;

            if maxP == minP
                cNorm = 0.5;
            else
                cNorm = (zones(i).price - minP) / (maxP - minP);
            end
            idxColor = max(1, min(256, round(cNorm*255)+1));

            h = patch(x, y, cmap(idxColor,:), ...
                'EdgeColor','k','FaceAlpha',0.75);

            zones(i).handle = h;
            set(h,'UserData',i);
            set(h,'ButtonDownFcn',@(src,event) zoneClicked_bin(src,event,zones,r,T,Nsim,cost_per_year));


            xc = mean(x);
            yc = mean(y);
            text(xc-10, yc-20, zones(i).name, ...
                'HorizontalAlignment','center', ...
                'FontWeight','bold', 'Color','w');
        end

end

colormap(cmap);
cb = colorbar;
cb.Label.String = 'Price of the derivative (€)';

hold off;


%% FUNCTIONS

% Prices and payoffs of proportional model
function [price, simLevels, simPayoff, M_payoff] = priceZoneDerivative...
    (z, N, r, T)
simLevels =  z.mu + z.sigma .*randn(N,1);
simPayoff = max(simLevels - z.strike, 0) * z.notional;
M_payoff = mean(simPayoff);
price = ((M_payoff * exp(-r*T)) / 10) * (1 + 0.40);
end

% Prices and payoffs of binomial model
function [price, simLevels, simPayoff, M_payoff] = ...
    priceZoneDerivative_bin (z, N, r, T)
simLevels = z.mu + z.sigma .* randn(N,1);
Levels = (simLevels > z.strike);
Levels_no_0 = Levels(Levels ~= 0);
Levels_80 = (simLevels > 80);
Levels_no_0_80 = Levels_80(Levels_80 ~= 0);
P = length(Levels_no_0) / length(Levels_no_0_80);
M_payoff = P * z.notional * 3;
price = ((M_payoff * exp(-r*T)) / 10) * (0.25 + 1);
simPayoff = M_payoff;
end





%% GRAPHS
% Graphs of the proportional model
function zoneClicked(src,~,zones,r,T,N,cost_per_year)
idx = get(src,'UserData');
z   = zones(idx);
fprintf('%s\n', z.name);
[price_loc, levels, payoff_loc, M_payoff_loc] = priceZoneDerivative(z, N, r, T);
figure(123); clf;
t = tiledlayout(1,3, 'TileSpacing','compact', 'Padding','compact');
sgtitle(['Analysis of the proportional derivative - ', z.name]);

% Histogram distribution per zone
ax1 = nexttile(1);
histogram(ax1, levels, 50);
xlabel(ax1, 'Water level (cm)');
ylabel(ax1, 'Frequency');
title(ax1, 'Distribution level');
grid(ax1, 'on');

% "short call" graph - insurance point of view
ax2 = nexttile(2);
w_min = z.strike - 10;
w_max = z.strike + 30;
w = linspace(w_min, w_max, 300);

premium = price_loc;
damages = max(w - z.strike, 0) * z.notional;
cost_curve   = -damages;              
profit_curve = premium - damages;     

% plot
plot(ax2, w, cost_curve, 'LineWidth', 2);      
hold(ax2, 'on');
plot(ax2, w, profit_curve, 'LineWidth', 2);
xline(ax2, z.strike, '--', 'LineWidth', 1.5);
yline(ax2, 0, ':');

% ---- Break-even point (profit = 0) ----
w_be = z.strike + premium / z.notional;
profit_be = 0;

plot(ax2, w_be, profit_be, 'o', ...
    'MarkerSize', 5, ...
    'MarkerFaceColor', [1 0.85 0], ...
    'MarkerEdgeColor', 'k');

% ---- Label con coordinate del break-even ----
label_be = sprintf('  (%.1f cm , %.0f €)', w_be, profit_be);
text(ax2, w_be, profit_be, label_be, ...
    'VerticalAlignment', 'bottom', ...
    'HorizontalAlignment', 'left', ...
    'FontSize', 9, ...
    'Color', 'w');
% -------------------------------------------

hold(ax2, 'off');


xlabel(ax2, 'Water level (cm)');
ylabel(ax2, '€');
title(ax2, 'Cost and profit from the insurance p.o.v.');
legend(ax2, {'Cost (claims)','Net profit','Strike','Zero line'}, ...
    'Location','southwest');
grid(ax2, 'on');

% zoom
xlim(ax2, [z.strike - 5, z.strike + 15]);
y_min = min([cost_curve, profit_curve]);
y_max = max([cost_curve, profit_curve]);
padding = 0.1 * (y_max - y_min);
ylim(ax2, [y_min - padding, y_max + padding]);

ax2.YAxis.Exponent = 0;
try
    ytickformat(ax2, '%,.0f');
catch
end

% Table with information of the derivative per zone
Price   = [zones.price]';
Payoff  = [zones.payoff_mean]';
Notional = [zones.notional]';
Zones_c = categorical({zones.name}');
Strike = [zones.strike]';
Total_cost = [ ...
    cost_per_year * (15/100), ...
    cost_per_year * (35/3)/100, ...
    cost_per_year * (35/3)/100, ...
    cost_per_year * (35/100), ...
    cost_per_year * (35/3)/100, ...
    cost_per_year * (15/3)/100 ]';

Quantity   = Total_cost ./ Payoff;
TOT_price  = Price  .* Quantity;
TOT_payoff = Payoff .* Quantity;

k = idx;
ax3 = nexttile(3);
pos = ax3.Position;
delete(ax3);

summaryData = {
    'Zone',        char(Zones_c(k));
    'Price (€)',   uint64(round(TOT_price(k), 2));
    'Payoff (€)',  uint64(round(TOT_payoff(k), 2));
    'Notional (€)',    uint64(round(Notional(k), 2));
    'Strike (cm)',     uint64(round(Strike(k), 2)); 
    };

uitable('Parent', gcf, ...
    'Data', summaryData, ...
    'ColumnName', {"Parameter","Value"}, ...
    'RowName', [], ...
    'Units', 'normalized', ...
    'Position', pos);


% ------------------------------------------------
figure(124); clf;
simulated_Levels = z.mu + z.sigma .* randn(N, 1);
minLevel_d = Strike(k);
mask = simulated_Levels >= minLevel_d;
simulated_Levels = simulated_Levels(mask);
Costs = simulated_Levels/minLevel_d * z.notional;
CI_costs = quantile(Costs, [0.05 0.95]);

nBins = 23;
edges = linspace(minLevel_d, max(simulated_Levels), nBins+1);
[binIdx, ~] = discretize(simulated_Levels, edges);

Level_Interval = strings(nBins,1);
Mean_Cost      = zeros(nBins,1);
Obs            = zeros(nBins,1);
P_L            = zeros(nBins,1);
Payoff           = zeros(nBins,1);

for i = 1:nBins
    Level_Interval(i) = sprintf('[%.1f , %.1f] cm', edges(i), edges(i+1));
    Mean_Cost(i)   = mean(Costs(binIdx == i)) + TOT_price(k);
    Obs(i)         = sum(binIdx == i);
    midLevel = (edges(i) + edges(i+1)) / 2;
    Payoff(i) = ((z.notional * (midLevel - minLevel_d)));
    P_L(i) = Payoff(i) -  Mean_Cost(i);
end


t = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
sgtitle(['Cost distribution and water-level intervals: ', z.name]);
ax1 = nexttile(1);
histogram(ax1, Costs, 50)
hold(ax1,'on')
xline(ax1, CI_costs(1), '--r', 'LineWidth', 2)
xline(ax1, CI_costs(2), '--r', 'LineWidth', 2)

xlabel(ax1,'Cost (€)')
ylabel(ax1,'Frequency')
title(ax1,'Distribution of costs')
legend(ax1, {'Histogram','5% quantile','95% quantile'}, ...
    'Location','best')
grid(ax1,'on')

ax1.XAxis.Exponent = 0;
ax1.XAxis.TickLabelFormat = '%,.0f';
hold(ax1,'off')

ax2 = nexttile(2);
pos = ax2.Position;
delete(ax2);   

summaryData = cell(nBins,3);
for i = 1:nBins
    summaryData{i,1} = char(Level_Interval(i));  
    summaryData{i,2} = Obs(i);
    summaryData{i,3} = uint64(round(Mean_Cost(i),0));
    summaryData{i,4} = uint64(round(Payoff(i),0));
end


uitable('Parent', gcf, ...
    'Data', summaryData, ...
    'ColumnName', { ...
        'Water level interval', ...
        'Observations', ...
        'Mean cost + price (€)', ...
        'Payoff (€)', ...
        }, ...
    'RowName', [], ...
    'Units','normalized', ...
    'Position', pos);

end




%% Graphs of the binomial model
function zoneClicked_bin(src,~,zones,r,T,N,cost_per_year)
idx = get(src,'UserData');
z   = zones(idx);
fprintf('\nClicked on: %s\n', z.name);
[price_loc, levels, payoff_loc, M_payoff_loc] = priceZoneDerivative_bin(z, N, r, T);
figure(123); clf;
t = tiledlayout(1,3, 'TileSpacing','compact', 'Padding','compact');
sgtitle(['Analysis of the binomial derivative - ', z.name]);

% Histogram distribution per zone
ax1 = nexttile(1);
histogram(ax1, levels, 50);
xlabel(ax1, 'Water level (cm)');
ylabel(ax1, 'Frequency');
title(ax1, 'Distribution level');
grid(ax1, 'on');


% "short call" graph - insurance point of view
ax2 = nexttile(2);
w_min = z.strike - 10;
w_max = z.strike + 30;
w = linspace(w_min, w_max, 300);

premium = price_loc;
damages = (w > z.strike) * z.notional;
cost_curve   = -damages;
profit_curve = premium - damages;

plot(ax2, w, cost_curve, 'LineWidth', 2);       
hold(ax2, 'on');
plot(ax2, w, profit_curve, 'LineWidth', 2);     
xline(ax2, z.strike, '--', 'LineWidth', 1.5);   
yline(ax2, 0, ':');                             

% ---- Break-even point (profit = 0) ----
w_be = z.strike;
profit_be = 0;
plot(ax2, w_be, profit_be, 'o', ...
    'MarkerSize', 5, ...
    'MarkerFaceColor', 'y', ...
    'MarkerEdgeColor', 'y');
% --------------------------------------


% ---- Label  ----------------------------
N_loss = (payoff_loc / (z.notional * 3));

label_be = sprintf('Quantity: %.2f\n', N_loss);

text(ax2, w_be, profit_be, label_be, ...
    'VerticalAlignment', 'bottom', ...
    'HorizontalAlignment', 'left', ...
    'FontSize', 9, ...
    'Color', 'w');
% ---------------------------------------------------------------

hold(ax2, 'off');

xlabel(ax2, 'Water level (cm)');
ylabel(ax2, '€');
title(ax2, 'Cost and profit from the insurance p.o.v. (binomial)');
legend(ax2, {'Cost (claims)','Net profit','Strike','Zero line','Break-even'}, ...
    'Location','southwest');
grid(ax2, 'on');

% zoom
xlim(ax2, [z.strike - 5, z.strike + 15]);
y_min = min([cost_curve, profit_curve]);
y_max = max([cost_curve, profit_curve]);
padding = 0.1 * (y_max - y_min);
ylim(ax2, [y_min - padding, y_max + padding]);

ax2.YAxis.Exponent = 0;
try
    ytickformat(ax2, '%,.0f');
catch
end


% Table with information of the derivative per zone
Price   = [zones.price]';
Payoff  = [zones.payoff_mean]';
Notional = [zones.notional]';
Zones_c = categorical({zones.name}');
Strike = [zones.strike]';
Total_cost = [ ...
    cost_per_year * (15/100), ...
    cost_per_year * (35/3)/100, ...
    cost_per_year * (35/3)/100, ...
    cost_per_year * (35/100), ...
    cost_per_year * (35/3)/100, ...
    cost_per_year * (15/3)/100 ]';

Quantity   = Total_cost ./ Payoff;
TOT_price  = Price  .* Quantity;
TOT_payoff = Payoff .* Quantity;

k = idx;
ax3 = nexttile(3);
pos = ax3.Position;
delete(ax3);

summaryData = {
    'Zone',        char(Zones_c(k));
    'Price (€)',   uint64(round(TOT_price(k), 2));
    'Payoff (€)',  uint64(round(TOT_payoff(k), 2));
    'Notional (€)',    uint64(round(Notional(k), 2));
    'Strike (cm)',     uint64(round(Strike(k), 2)); 
    };

uitable('Parent', gcf, ...
    'Data', summaryData, ...
    'ColumnName', {"Parameter","Value"}, ...
    'RowName', [], ...
    'Units', 'normalized', ...
    'Position', pos);



% --------------------------------------------------
figure(124); clf;
simulated_Levels = z.mu + z.sigma .* randn(N, 1);
minLevel_d = Strike(k);
mask = simulated_Levels >= minLevel_d;
simulated_Levels = simulated_Levels(mask);
Costs = simulated_Levels/minLevel_d * z.notional;
CI_costs = quantile(Costs, [0.05 0.95]);


nBins = 10;
edges = linspace(minLevel_d, max(simulated_Levels), nBins+1);
[binIdx, ~] = discretize(simulated_Levels, edges);
edges_2 = linspace(0, 1, nBins+1);

Interval       = strings(nBins,1);
Mean_Cost      = zeros(nBins,1);
Payoff         = zeros(nBins,1);
Level_Interval = strings(nBins,1);

for i = 1:nBins
    Level_Interval(i) = sprintf('[%.1f , %.1f] %', edges(i), edges(i+1));
    Mean_Cost(i)   = mean(Costs(binIdx == i)) + TOT_price(k);
    Interval(i) = sprintf('[%.1f , %.1f] percentage', edges_2(i), edges_2(i+1));
    P = (edges_2(i) + edges_2(i+1)) / 2;
    Payoff(i) = ((z.notional * P)) * 4;
end


t = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
sgtitle(['Cost distribution and water-level intervals: ', z.name]);
ax1 = nexttile(1);
histogram(ax1, Costs, 50)
hold(ax1,'on')
xline(ax1, CI_costs(1), '--r', 'LineWidth', 2)
xline(ax1, CI_costs(2), '--r', 'LineWidth', 2)

xlabel(ax1,'Cost (€)')
ylabel(ax1,'Frequency')
title(ax1,'Distribution of costs')
legend(ax1, {'Histogram','5% quantile','95% quantile'}, ...
    'Location','best')
grid(ax1,'on')

ax1.XAxis.Exponent = 0;
ax1.XAxis.TickLabelFormat = '%,.0f';
hold(ax1,'off')

ax2 = nexttile(2);
pos = ax2.Position;
delete(ax2);   

summaryData = cell(nBins,3);
for i = 1:nBins
    summaryData{i,1} = char(Interval(i));  
    summaryData{i,2} = uint64(round(Mean_Cost(i),0));
    summaryData{i,3} = uint64(round(Payoff(i),0));
end


uitable('Parent', gcf, ...
    'Data', summaryData, ...
    'ColumnName', { ...
        'Interval', ...
        'Mean cost + price (€)', ...
        'Payoff (€)', ...
        }, ...
    'RowName', [], ...
    'Units','normalized', ...
    'Position', pos);

end

table()