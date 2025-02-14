import torch
import torch.nn.functional as F

def euler2mat(angle):
    """Convert euler angles to rotation matrix"""
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).view(B, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    return rot_mat

def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def pose_vec2mat(vec, mode='euler'):
    """Convert Euler parameters to transformation matrix."""
    if mode is None:
        return vec
    trans, rot = vec[:, :3].unsqueeze(-1), vec[:, 3:]
    if mode == 'euler':
        rot_mat = euler2mat(rot)
    elif mode == 'axis_angle':
        rot_mat = axis_angle_to_matrix(rot)
    else:
        raise ValueError('Rotation mode not supported {}'.format(mode))
    mat = torch.cat([rot_mat, trans], dim=2)  # [B,3,4]
    return mat

def invert_pose(T):
    """Inverts a [B,4,4] torch.tensor pose"""
    Tinv = torch.eye(4, device=T.device, dtype=T.dtype).repeat([len(T), 1, 1])
    Tinv[:, :3, :3] = torch.transpose(T[:, :3, :3], -2, -1)
    Tinv[:, :3, -1] = torch.bmm(-1. * Tinv[:, :3, :3], T[:, :3, -1].unsqueeze(-1)).squeeze(-1)
    return Tinv

class Pose:
    """
    Pose class, that encapsulates a [4,4] transformation matrix
    for a specific reference frame
    """
    def __init__(self, mat):
        """
        Initializes a Pose object.

        Parameters
        ----------
        mat : torch.Tensor [B,4,4]
            Transformation matrix
        """
        assert tuple(mat.shape[-2:]) == (4, 4)
        if mat.dim() == 2:
            mat = mat.unsqueeze(0)
        assert mat.dim() == 3
        self.mat = mat

    def __len__(self):
        """Batch size of the transformation matrix"""
        return len(self.mat)

########################################################################################################################

    @classmethod
    def identity(cls, N=1, device=None, dtype=torch.float):
        """Initializes as a [4,4] identity matrix"""
        return cls(torch.eye(4, device=device, dtype=dtype).repeat([N,1,1]))
    
    @classmethod
    def make_4x4(cls, Tcw, device=None, dtype=torch.float):
        """
        Converts a 3x4 transformation matrix to a 4x4 transformation matrix.

        Parameters
        ----------
        Tcw : torch.Tensor [B, 3, 4]
            The input transformation matrix.

        Returns
        -------
        Tcw_4x4 : torch.Tensor [B, 4, 4]
            The output 4x4 transformation matrix.
        """
        if Tcw.shape[1:] == (4, 4):  # If already 4x4, no changes needed
            return cls(Tcw)

        B = Tcw.shape[0]
        bottom_row = torch.tensor([0, 0, 0, 1], device=Tcw.device, dtype=Tcw.dtype).unsqueeze(0).repeat(B, 1, 1)
        Tcw_4x4 = torch.cat([Tcw, bottom_row], dim=1)  # Add bottom row to make it 4x4
        return Tcw_4x4 #cls(Tcw_4x4)

    @classmethod
    def from_vec(cls, vec, mode):
        """Initializes from a [B,6] batch vector"""
        mat = pose_vec2mat(vec, mode)  # [B,3,4]
        pose = torch.eye(4, device=vec.device, dtype=vec.dtype).repeat([len(vec), 1, 1])
        pose[:, :3, :3] = mat[:, :3, :3]
        pose[:, :3, -1] = mat[:, :3, -1]
        return cls(pose)

########################################################################################################################

    @property
    def shape(self):
        """Returns the transformation matrix shape"""
        return self.mat.shape

    def item(self):
        """Returns the transformation matrix"""
        return self.mat

    def repeat(self, *args, **kwargs):
        """Repeats the transformation matrix multiple times"""
        self.mat = self.mat.repeat(*args, **kwargs)
        return self

    def inverse(self):
        """Returns a new Pose that is the inverse of this one"""
        return Pose(invert_pose(self.mat))

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.mat = self.mat.to(*args, **kwargs)
        return self

########################################################################################################################

    def transform_pose(self, pose):
        """Creates a new pose object that compounds this and another one (self * pose)"""
        assert tuple(pose.shape[-2:]) == (4, 4)
        return Pose(self.mat.bmm(pose.item()))

    def transform_points(self, points):
        """Transforms 3D points using this object"""
        assert points.shape[1] == 3
        B, _, H, W = points.shape
        out = self.mat[:,:3,:3].bmm(points.view(B, 3, -1)) + \
              self.mat[:,:3,-1].unsqueeze(-1)
        return out.view(B, 3, H, W)

    def __matmul__(self, other):
        """Transforms the input (Pose or 3D points) using this object"""
        if isinstance(other, Pose):
            return self.transform_pose(other)
        elif isinstance(other, torch.Tensor):
            if other.shape[1] == 3 and other.dim() > 2:
                assert other.dim() == 3 or other.dim() == 4
                return self.transform_points(other)
            else:
                raise ValueError('Unknown tensor dimensions {}'.format(other.shape))
        else:
            raise NotImplementedError()
        
        
# #test quat2rot
# import numpy as np
# matrix = np.array([
#     [0.9878559, -0.1388341, 0.0697565],
#     [0.1551003,  0.9077532, -0.3897793],
#     [-0.0092070,  0.3958650,  0.9182625]
# ])

# mat = torch.from_numpy(matrix).float()

# from pytorch3d.transforms import matrix_to_quaternion as torch_m2q
# quat = torch_m2q(mat) #[w, x, y, z]
# print(quat)

# quat_test = matrix_to_quaternion(mat)
# print(quat_test)
