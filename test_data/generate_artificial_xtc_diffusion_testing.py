'''Python module to generate artificial xtc data for unit testing of mean_square_displacement_by_species() diffusion analysis code.'''

import MDAnalysis
import numpy
import scipy
import MDAnalysis.coordinates.XTC
import MDAnalysis.coordinates.GRO

def main():
    #load a dummy coordinate file that contains only three CG residues (MET, ARG, CYS -- in that order):
    dummy_universe = MDAnalysis.Universe('dummy.gro')

    #produce selections / AtomGroups for each residue:
    list_residue_objects = dummy_universe.selectAtoms('all').residues
    MET_selection, ARG_selection, CYS_selection = list_residue_objects

    #set all atoms in all residues to the original initially:
    for residue in [MET_selection, ARG_selection, CYS_selection]: 
        atoms_in_residue = residue.numberOfAtoms()
        origin_coordinate_array = numpy.zeros((atoms_in_residue,3))
        residue.set_positions(origin_coordinate_array)

    xtc_writer = MDAnalysis.coordinates.XTC.XTCWriter('diffusion_testing.xtc',7)

    MET_base_array = numpy.zeros((2,3))
    ARG_base_array = numpy.zeros((3,3))
    CYS_base_array = numpy.zeros((2,3))
    

    for frame_number in range(1,21):
        #increment MET atoms in the X direction by 0.1 A per frame
        MET_base_array = numpy.column_stack((MET_base_array[...,0] + 0.1,MET_base_array[...,1:]))
        MET_selection.set_positions(MET_base_array)
        #increment ARG atoms in the X direction by 1.2 A per frame
        ARG_base_array = numpy.column_stack((ARG_base_array[...,0] + 1.2,ARG_base_array[...,1:]))
        ARG_selection.set_positions(ARG_base_array)
        #increment CYS atoms in the X direction by 4.0 A per frame
        CYS_base_array = numpy.column_stack((CYS_base_array[...,0] + 4.0,CYS_base_array[...,1:]))
        CYS_selection.set_positions(CYS_base_array)
        xtc_writer.write(dummy_universe)

    #should be a 20 frame trajectory with the above incremental movements per frame

if __name__ == '__main__':
    pass



