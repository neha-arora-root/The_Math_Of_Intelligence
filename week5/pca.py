import numpy as np

class PCA(object):
    '''
    For a dataset with n samples and m dimensions, PCA (Principal Component Analysis) produces 
    an orthogonally arranged set of vectors that try to capture maximum variance within the data.
    There could be two implementation using eigenvalues and vectors and the other using SVD 
    (Singular Value Decomposition).
    '''
    
    def __init__ (self, X, svd=True):
        
        self.N, self.dim, *rest = X.shape
        self.X = X
        if svd is True:
            self.cell = "svd"
        else:
            self.cell = "eigen"
        
        '''
        Implementation using Singular Value Decomposition is less computationally expensive 
        and hence used as the default method.
        U S V' = svd(X) and can be obtained as follows:
        [U, S, Vt] = svd(X)
        U: Left singular vectors
        S: Contains singular values on its diagonal
        V: Right singular vectors
        
        U and V have this amazing property that U and V are unitary matrices which implies:
        UU' = I, and 
        VV' = I
        
        The loading T is calculated as:
        T = XV (V is identical to W calculated with eigenvalues method)
        Therefore, 
        T = XV = USV'V = USI = US
        '''
        if svd:
            X_std = X
        else:
            X_std = (X - np.mean(X, axis=0))/(np.std(X, axis=0)+1e-13)
        if (svd):
            print ("Implementation with SVD")
            [self.U, self.s, self.Vt] = np.linalg.svd(X_std)
            self.V = self.Vt.T
            # Converting s to a matrix
            self.S = np.zeros((self.N, self.dim))
            self.S[:self.dim, :self.dim] = np.diag(self.s)
            self.variance_ratio = self.s
            
        else:
            print ("Implementation with eigenvalues")
            
            cov_mat = np.cov(X_std, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
            eigen_pairs = [(eigenvalues[idx], eigenvectors[:,idx]) for idx in range(len(eigenvalues))]
            eigen_pairs.sort()
            eigen_pairs.reverse()
            eigenvalues_sorted = [eigen_pairs[idx][0] for idx in range(len(eigenvalues))]
            eigenvectors_sorted = [eigen_pairs[idx][1] for idx in range(len(eigenvalues))]
            self.variance_ratio = eigenvectors_sorted
            
            self.eigenvals = eigenvalues_sorted
            self.eigenvecs = self.V = eigenvectors_sorted
            
            
    def variance_explained_ratio (self):
            
        '''
        Returns the cumulative variance captured with each added principal component
        '''
        return np.cumsum(self.variance_ratio)/np.sum(self.variance_ratio)
            
    def X_projected (self, dataset, r):
            
        '''
        Returns the data X projected along the first r principal components
        '''
            
        if r is None:
            r = self.dim
        X_proj = np.zeros((r, self.N))
        P_reduce = self.V[:,0:r]
        X_proj = dataset.dot(P_reduce)
        return X_proj
            
            
            
            
            
        
        