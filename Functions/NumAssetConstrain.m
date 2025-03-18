function [c,ceq] = NumAssetConstrain(x, minNumAsset, maxNumAsset)
    numSelectedAssets = sum(x > 1e-5);
    
    c1= minNumAsset - numSelectedAssets;
    c2 = numSelectedAssets - maxNumAsset;
    c = [c1;c2];
    ceq = [];
end