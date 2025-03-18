function [] = show_pie(weights, name)


categories = {'InformationTechnology', 'Financials', 'HealthCare', 'ConsumerDiscretionary', 'CommunicationServices', ...
              'Industrials', 'ConsumerStaples', 'Energy', 'Utilities', 'RealEstate', ...
              'Materials', 'Momentum', 'Value', 'Growth', 'Quality', 'LowVolatility'};

% Remove Assets with Zero Percentage
nonZeroIndices = weights > 1e-5;
categories = categories(nonZeroIndices);
w = weights(nonZeroIndices);

% Threshold for Grouping Small Assets
threshold = 0.02; 

% Group Small Assets
smallAssetIndices = w < threshold; % Find small assets
largeAssetIndices = ~smallAssetIndices;      % Find large assets
categoriesLarge = categories(largeAssetIndices); % Keep large assets
percentagesLarge = w(largeAssetIndices);

% Add "Others" category
if any(smallAssetIndices) % If there are any small assets
    categoriesLarge{end+1} = 'Others'; % Add "Others" label
    percentagesLarge(end+1) = sum(w(smallAssetIndices)); % Sum of small percentages
end

% Update variables
categories = categoriesLarge;
w = percentagesLarge;

% Generate a Custom Colormap with Enough Unique Colors
numAssets = length(categories);
customColors = parula(numAssets); % hsv creates a smooth transition of unique colors

% Create the Pie Chart
figure;
p = pie(w);

hold on;

% Overlay a White Circle to Create the Doughnut Effect
theta = linspace(0, 2*pi, 100);
x = 0.6 * cos(theta); % Adjust the 0.5 factor to control the size of the void
y = 0.6 * sin(theta);
fill(x, y, 'w', 'EdgeColor', 'none'); % Fill with white and no edge

% Apply Custom Colors to the Pie Slices
for i = 1:2:length(p) % p(1), p(3), etc., are pie slices
    p(i).FaceColor = customColors((i+1)/2, :);
end

% Set Legend and Aesthetic Adjustments
legend(categories, 'Location', 'eastoutside', 'FontSize', 10); % Adjust size for readability
title([name, 'composition'], 'FontSize', 16, 'FontWeight', 'bold');
set(gca, 'Color', [0.95 0.95 0.95]); % Light gray background
axis equal; % Ensure the chart remains circular

end