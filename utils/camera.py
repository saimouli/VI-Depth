import torch
import torch.nn as nn
from utils.pose import Pose
from pytorch3d.transforms import se3_exp_map, se3_log_map

def scale_intrinsics(K, x_scale, y_scale):
    """Scale intrinsics given x_scale and y_scale factors"""
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
    return K

def meshgrid(B, H, W, dtype, device, normalized=False):
    """
    Create meshgrid with a specific resolution

    Parameters
    ----------
    B : int
        Batch size
    H : int
        Height size
    W : int
        Width size
    dtype : torch.dtype
        Meshgrid type
    device : torch.device
        Meshgrid device
    normalized : bool
        True if grid is normalized between -1 and 1

    Returns
    -------
    xs : torch.Tensor [B,1,W]
        Meshgrid in dimension x
    ys : torch.Tensor [B,H,1]
        Meshgrid in dimension y
    """
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    ys, xs = torch.meshgrid([ys, xs])
    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])

def image_grid(B, H, W, dtype, device, normalized=False):
    """
    Create an image grid with a specific resolution

    Parameters
    ----------
    B : int
        Batch size
    H : int
        Height size
    W : int
        Width size
    dtype : torch.dtype
        Meshgrid type
    device : torch.device
        Meshgrid device
    normalized : bool
        True if grid is normalized between -1 and 1

    Returns
    -------
    grid : torch.Tensor [B,3,H,W]
        Image grid containing a meshgrid in x, y and 1
    """
    xs, ys = meshgrid(B, H, W, dtype, device, normalized=normalized)
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=1)
    return grid

#Convert initial poses to pyproj.LieTensor (4x4 SE3 matrices)
def pose_to_se3(pose):
    """
    Convert [B, 4, 4] pose matrix to [B, 6] se3 parameters (translation + axis-angle).
    Note: pytorch takes [R 0;T 1] format
    """
    # Extract rotation and translation from standard 3x4 pose matrix
    rotation = pose[:,:3, :3]  # [B, 3, 3]
    translation = pose[:,:3,3]  # [B, 3]

    # Build 4x4 matrix compatible with PyTorch3D's se3_log_map
    batch_size = rotation.shape[0]
    transform = torch.eye(4, device=rotation.device).repeat(batch_size, 1, 1)
    transform[:, :3, :3] = rotation  # Upper-left 3x3 = rotation
    transform[:, :3, 3] = 0.0        # Zero-out 4th column (PyTorch3D requirement)
    transform[:, 3, :3] = translation  # Translation in 4th row[:3]

    # Convert to se3 parameters (translation + axis-angle)
    return se3_log_map(transform)  # [B, 6]
    
# Convert back to 3x4 matrix format if needed
def se3_to_pose(se3_pose):
    """Convert pyproj.LieTensor to [B, 3, 4] pose matrix"""
    transform = se3_exp_map(se3_pose)   # [B, 4, 4]
    rotation = transform[:, :3, :3]
    translation = transform[:, 3, :3]
        
    bottom_row = torch.tensor([0, 0, 0, 1], 
                            device=rotation.device, dtype=rotation.dtype).repeat(rotation.shape[0], 1, 1)
    mat = torch.cat([torch.cat([rotation, translation.unsqueeze(-1)], dim=-1), bottom_row], dim=1)
    return mat  # [B, 4, 4]

def se3_update(current_se3, delta_se3):
    """Update SE3 parameters using PyTorch3D's exponential map"""
    # Convert to 4x4 transformation matrices
    current_mat = se3_exp_map(current_se3)  # [B, 4, 4]
    delta_mat = se3_exp_map(delta_se3)      # [B, 4, 4]
        
    # Rearrange PyTorch3D format [R 0; T 1] to standard format [R T; 0 1]
    current_mat_standard = current_mat.clone()
    current_mat_standard[:, :3, 3] = current_mat[:, 3, :3]  # Move translation to correct position
    current_mat_standard[:, 3, :3] = 0  # Ensure bottom row is [0, 0, 0, 1]
        
    delta_mat_standard = delta_mat.clone()
    delta_mat_standard[:, :3, 3] = delta_mat[:, 3, :3]  # Move translation to correct position
    delta_mat_standard[:, 3, :3] = 0  # Ensure bottom row is [0, 0, 0, 1]
        
    # Compose transformations in standard format
    #Apply update: new_T = delta_T * T  (left multiplication)
    updated_mat = delta_mat_standard @ current_mat_standard
        
    # Convert back to se3 parameters
    return pose_to_se3(updated_mat)  # [B, 6]
    
class Camera(nn.Module):
    def __init__(self, K, Twc=None):
        super().__init__()
        self.K = K
        self.Twc = Pose.identity(len(K)) if Twc is None else Pose.make_4x4(Twc)

    def __len__(self):
        """Batch size of the camera intrinsics"""
        return len(self.K)
    
    def to(self, *args, **kwargs):
        self.K = self.K.to(*args, **kwargs)
        self.Twc = self.Twc.to(*args, **kwargs)
        return self
    
    @property
    def fx(self):
        """Focal length in x"""
        return self.K[:, 0, 0]

    @property
    def fy(self):
        """Focal length in y"""
        return self.K[:, 1, 1]

    @property
    def cx(self):
        """Principal point in x"""
        return self.K[:, 0, 2]

    @property
    def cy(self):
        """Principal point in y"""
        return self.K[:, 1, 2]
    
    @property
    def Tcw(self):
        """World -> Camera pose transformation (inverse of Tcw)"""
        return self.Twc.inverse()

    @property
    def Kinv(self):
        """Inverse intrinsics (for lifting)"""
        Kinv = self.K.clone()
        Kinv[:, 0, 0] = 1. / self.fx
        Kinv[:, 1, 1] = 1. / self.fy
        Kinv[:, 0, 2] = -1. * self.cx / self.fx
        Kinv[:, 1, 2] = -1. * self.cy / self.fy
        return Kinv

    def scaled(self, x_scale, y_scale=None):
        """
        Returns a scaled version of the camera (changing intrinsics)

        Parameters
        ----------
        x_scale : float
            Resize scale in x
        y_scale : float
            Resize scale in y. If None, use the same as x_scale

        Returns
        -------
        camera : Camera
            Scaled version of the current cmaera
        """
        # If single value is provided, use for both dimensions
        if y_scale is None:
            y_scale = x_scale
        # If no scaling is necessary, return same camera
        if x_scale == 1. and y_scale == 1.:
            return self
        # Scale intrinsics and return new camera with same Pose
        K = scale_intrinsics(self.K.clone(), x_scale, y_scale)
        return Camera(K, Twc=self.Twc)
    
    def reconstruct(self, depth, frame='w'):
        """
        Reconstructs pixel-wise 3D points from a depth map.

        Parameters
        ----------
        depth : torch.Tensor [B,1,H,W]
            Depth map for the camera
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.tensor [B,3,H,W]
            Pixel-wise 3D points
        """
        if len(depth.shape) == 3:
            depth = depth.unsqueeze(1)
            
        B, C, H, W = depth.shape
        assert C == 1

        # Create flat index grid
        grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False)  # [B,3,H,W]
        flat_grid = grid.view(B, 3, -1)  # [B,3,HW]

        # Estimate the outward rays in the camera frame
        xnorm = (self.Kinv.bmm(flat_grid)).view(B, 3, H, W)
        # Scale rays to metric depth
        Xc = xnorm * depth

        # If in camera frame of reference
        if frame == 'c':
            return Xc
        # If in world frame of reference
        elif frame == 'w':
            if not isinstance(self.Twc, Pose):  
                return Pose(self.Twc) @ Xc  
            else:  
                return self.Twc @ Xc  

        # If none of the above
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

    def project(self, X, frame='w', normalize=True):
        """
        Projects 3D points onto the image plane

        Parameters
        ----------
        X : torch.Tensor [B,3,H,W]
            3D points to be projected
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.Tensor [B,H,W,2]
            2D projected points that are within the image boundaries
        """
        B, C, H, W = X.shape
        assert C == 3

        # Project 3D points onto the camera image plane
        if frame == 'c':
            Xc = self.K.bmm(X.view(B, 3, -1))
        elif frame == 'w':
            if not isinstance(self.Tcw, Pose):
                Xc = self.K.bmm((Pose(self.Tcw).to(X.dtype) @ X).view(B, 3, -1))
            else:
                Xc = self.K.bmm((self.Tcw.to(X.dtype) @ X).view(B, 3, -1))
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

        # Normalize points
        X = Xc[:, 0]
        Y = Xc[:, 1]
        Z = Xc[:, 2].clamp(min=1e-5)
        if normalize:
            Xnorm = 2 * (X / Z) / (W - 1) - 1.  #(-1, 1)
            Ynorm = 2 * (Y / Z) / (H - 1) - 1.
        else:
            Xnorm = X / Z
            Ynorm = Y / Z

        # Clamp out-of-bounds pixels
        # Xmask = ((Xnorm > 1) + (Xnorm < -1)).detach()
        # Xnorm[Xmask] = 2.
        # Ymask = ((Ynorm > 1) + (Ynorm < -1)).detach()
        # Ynorm[Ymask] = 2.

        # Return pixel coordinates
        return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)