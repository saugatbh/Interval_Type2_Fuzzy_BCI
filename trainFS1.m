function obj=trainFS1(X,lab,nDays)
% This function trains the IT2FS classifier using the observations (rows)
% in X and the corresponding labels in lab i.e. X and lab together
% constitutes the train-set. This function should be used in conjunction 
% with paraGMF1.m.
% Created by Saugat Bhattacharyya. Last updated on 27/11/2013.
% Syntax: obj = predictFS1(X, lab, nDays)
% Input:
% 1. X: It is a matrix, whose rows represents the train observations and
% columns represent the features. Let it be of dimension N x d.
% 2. lab: It is a column vector of length N identifying the class of each
% observation of X.
% 3. nDays: It is a scalar value representing the number of sessions or the
% sets contributing to the formation of UMF and LMF for IT2FS.
% Note: X and lab for nDays should be vertically concatenated.
% Output:
% 1. obj: It is a structure representing the trained IT2FS classifier. It
% stores the associated information in its constituents.
obj.f=X; % storing the train-set in obj
obj.targets=lab; % storing the labesl in obj
obj.nFeatures=size(obj.f,2); % storing the feature dimension in obj
obj.Classes=unique(lab); % storing the different distinct labels in obj
obj.nClasses=size(obj.Classes,1); % storing number of classes in obj
obj.nSets=nDays; % storing the value for number of sets in obj

% calling paraGMF1 to obtain the parameters (mean and std dev) for each 
% set, for each feature dimension and for each class to construct the
% Gaussian Membership Function
[obj.m,obj.sd]=paraGMF1(obj.f,obj.targets,obj.Classes,obj.nSets,obj.nClasses,obj.nFeatures);

obj.spanMF=1000;% resolution of gaussian curve, stored in obj

% initializing variable for storing values yielded by Gaussian MF
obj.MF=zeros(obj.nClasses,obj.nFeatures,obj.spanMF,obj.nSets);

for i=1:obj.nClasses
    for j=1:obj.nFeatures
        % create a row vector x which has #obj.spanMF uniformly 
        % distributed values between the lowest and the highest value 
        % observed along the j-th feature dimension       
        x=linspace(floor(min(obj.f(:,j))),ceil(max(obj.f(:,j))),obj.spanMF);
        for l=1:obj.nSets
            % for i-th class, j-th feature dimension and l-th set, the
            % membership values for each point in x is calculated depending
            % on the parameters corresponding to i-th class, j-th feature
            % dimension and l-th set in the training set
            obj.MF(i,j,:,l)=gaussmf(x,[obj.sd(i,j,l) obj.m(i,j,l)]);
        end
        % for i-th class and j-th feature dimension the maximum and minimum
        % of all the sets (along the 4th dimension) is chosen to yield the
        % UMF and LMF, respectively which are also stored in obj.
        obj.UMF(i,j,:)=max(obj.MF(i,j,:,:),[],4);
        obj.LMF(i,j,:)=min(obj.MF(i,j,:,:),[],4);
    end
end
