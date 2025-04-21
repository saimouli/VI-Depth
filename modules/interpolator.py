import numpy as np
np.set_printoptions(suppress=True)

from scipy.interpolate import griddata, Rbf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

def polynomial_interpolate_map(map_size, knot_coords, knot_values, degree=3, reg_lambda=1.0):
    """
    Create a polynomial interpolation of sparse points as scaffolding for depth refinement.
    
    Args:
        map_size: Tuple of (height, width) for the output map
        knot_coords: Array of shape (2, N) with coordinates of known points (x, y)
        knot_values: Array of shape (N,) with values at known points
        degree: Degree of polynomial (2=quadratic, 3=cubic)
        reg_lambda: Regularization strength for Ridge regression
        
    Returns:
        Interpolated map of shape map_size
    """
    # Create coordinate grid for the entire map
    grid_y, grid_x = np.mgrid[0:map_size[0], 0:map_size[1]]
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T  # Shape: (H*W, 2)
    
    # Prepare knot coordinates for scikit-learn (transpose from (2, N) to (N, 2))
    knot_coords_T = knot_coords.T  # Shape: (N, 2)
    
    # Print info about the points
    n_points = knot_coords.shape[1]
    #print(f"Fitting polynomial of degree {degree} with {n_points} points")
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    
    # Create ridge regression model with regularization to prevent overfitting
    model = make_pipeline(
        poly_features,
        Ridge(alpha=reg_lambda)
    )
    
    # Fit model on known points
    model.fit(knot_coords_T, knot_values)
    
    # Predict values for entire grid
    grid_values = model.predict(grid_coords)
    
    # Reshape to map size
    interpolated_map = grid_values.reshape(map_size)
    
    # Clamp to reasonable range based on known values
    min_val, max_val = np.percentile(knot_values, [2.5, 97.5])
    margin = (max_val - min_val) * 0.20  # Allow 5% extension beyond observed range
    
    # Add margin to the range
    valid_min = max(0.1, min_val - margin)  # Don't go below 0.1 (assuming scale factors)
    valid_max = max_val + margin  # Upper bound with margin
    
    # Clamp values
    interpolated_map = np.clip(interpolated_map, valid_min, valid_max)
    
    return interpolated_map.astype(np.float32)


class PolynomialInterpolator2D(object):
    def __init__(self, pred_inv, sparse_depth_inv, valid):
        """
        Polynomial interpolator for scale maps with 180-220 sparse points.
        
        Args:
            pred_inv: Predicted inverse depth map
            sparse_depth_inv: Sparse inverse depth map
            valid: Validity map showing where sparse points exist
        """
        self.pred_inv = pred_inv
        self.sparse_depth_inv = sparse_depth_inv
        self.valid = valid
        
        self.map_size = np.shape(pred_inv)
        self.num_knots = np.sum(valid)
        nonzero_y_loc = np.nonzero(valid)[0]
        nonzero_x_loc = np.nonzero(valid)[1]
        self.knot_coords = np.stack((nonzero_x_loc, nonzero_y_loc))
        
        # Calculate scale values at knot points
        self.knot_scales = sparse_depth_inv[valid] / pred_inv[valid]
        
        self.knot_list = []
        for i in range(self.num_knots):
            self.knot_list.append((int(self.knot_coords[0,i]), int(self.knot_coords[1,i])))
        
        # to be computed
        self.interpolated_scale_map = None
        self.output = None

    def generate_polynomial_scale_map(self, degree=3, reg_lambda=1.0, fallback=True):
        """
        Generate interpolated scale map using polynomial regression.
        
        Args:
            degree: Polynomial degree (2=quadratic, 3=cubic)
            reg_lambda: Regularization strength
            fallback: Whether to fall back to linear interpolation if polynomial fails
            
        Returns:
            Interpolated scale map
        """
        try:
            self.interpolated_scale_map = polynomial_interpolate_map(
                map_size=self.map_size,
                knot_coords=self.knot_coords,
                knot_values=self.knot_scales,
                degree=degree,
                reg_lambda=reg_lambda
            )
        except Exception as e:
            print(f"Polynomial interpolation failed: {e}")
            if fallback:
                print("Falling back to linear interpolation")
                grid_y, grid_x = np.mgrid[0:self.map_size[0], 0:self.map_size[1]]
                self.interpolated_scale_map = griddata(
                    points=self.knot_coords.T,
                    values=self.knot_scales,
                    xi=(grid_y, grid_x),
                    method='linear',
                    fill_value=1.0
                ).astype(np.float32)
            else:
                raise
            
        return self.interpolated_scale_map
    
    def apply_scale_map(self):
        """
        Apply the interpolated scale map to the prediction.
        
        Returns:
            Scaled inverse depth map
        """
        if self.interpolated_scale_map is None:
            self.generate_polynomial_scale_map()
            
        # Apply scale map with original values preserved at knot points
        scaled_inv_depth = self.pred_inv * self.interpolated_scale_map
        
        # Ensure knot points match exactly
        scaled_inv_depth[self.valid] = self.sparse_depth_inv[self.valid]
            
        self.output = scaled_inv_depth
        return scaled_inv_depth
    
def interpolate_knots(map_size, knot_coords, knot_values, interpolate, fill_corners, degree=3):
    grid_x, grid_y = np.mgrid[0:map_size[0], 0:map_size[1]]

    if interpolate == 'rbf':
        # Extract x and y coordinates
        x = knot_coords[0]
        y = knot_coords[1]
        
        # Create RBF interpolator
        # 'thin_plate' is generally good for spatial data with few points
        rbf = Rbf(x, y, knot_values, function='thin_plate', smooth=0.03)
        
        # Apply RBF interpolation to the entire grid
        yy, xx = np.meshgrid(np.arange(map_size[0]), np.arange(map_size[1]), indexing='ij')
        interpolated_map = rbf(xx, yy)
        
        return interpolated_map
    else:
        # Original griddata method for other interpolation types
        interpolated_map = griddata(
            points=knot_coords.T,
            values=knot_values,
            xi=(grid_y, grid_x),
            method=interpolate,
            fill_value=1.0)
    
    # elif interpolate == 'polynomial':
    #     degree = degree,
    #     interpolated_map = polynomial_interpolate_map(
    #         map_size=map_size,
    #         knot_coords=knot_coords,
    #         knot_values=knot_values,
    #         degree=degree,
    #         reg_lambda=1.0
    #     )
        
        return interpolated_map


class Interpolator2D(object):
    def __init__(self, pred_inv, sparse_depth_inv, valid):
        self.pred_inv = pred_inv
        self.sparse_depth_inv = sparse_depth_inv
        self.valid = valid

        self.map_size = np.shape(pred_inv)
        self.num_knots = np.sum(valid)
        nonzero_y_loc = np.nonzero(valid)[0]
        nonzero_x_loc = np.nonzero(valid)[1]
        self.knot_coords = np.stack((nonzero_x_loc, nonzero_y_loc))
        self.knot_scales = sparse_depth_inv[valid] / pred_inv[valid]
        self.knot_shifts = sparse_depth_inv[valid] - pred_inv[valid]

        self.knot_list = []
        for i in range(self.num_knots):
            self.knot_list.append((int(self.knot_coords[0,i]), int(self.knot_coords[1,i])))

        # to be computed
        self.interpolated_scale_map = None
        self.confidence_map = None
        self.output = None

    def generate_interpolated_scale_map(self, interpolate_method, fill_corners=False):
        self.interpolated_scale_map = interpolate_knots(
            map_size=self.map_size, 
            knot_coords=self.knot_coords, 
            knot_values=self.knot_scales,
            interpolate=interpolate_method,
            fill_corners=fill_corners
        ).astype(np.float32)