clear variables %#ok<*NASGU>
addpath([fileparts(fileparts(which('braph2'))) filesep 'NetworkModelling'])

%% Visulaize Brain Atlas
im_ba = ImporterBrainAtlasXLS( ...
    'FILE', [fileparts(fileparts(which('braph2'))) filesep 'NetworkModelling' filesep 'data' filesep 'bna_atlas.xlsx'], ...
    'WAITBAR', true ...
    );

ba = im_ba.get('BA');
bapf = BrainAtlasPF('BA', ba);
gui_bapf = GUIFig('PF', bapf);
gui_bapf.get('DRAW');
gui_bapf.get('SHOW')

bapf.get('ST_SURFACE').set('FACEALPHA', 0.6);
bapf.get('ST_AMBIENT').set('CAMLIGHT', 'headlight');
bapf.get('ST_AMBIENT').set('SHADING', 'interp');
bapf.get('ST_AMBIENT').set('COLORMAP', 'white');
bapf.get('ST_AMBIENT').set('COLORMAP', 'none');
bapf.set('SPHS', false);
bapf.set('SPHS', true);
contenttype = 'vector'; resolution = 300; colorspace = 'rgb';
bapf.set('VIEW', BrainSurfacePF.VIEW_SL_AZEL)
braph2print(gui_bapf.get('PF').get('H'), [fileparts(fileparts(which('braph2'))) filesep 'NetworkModelling' filesep 'atlas_SL.jpg'], 'ContentType', contenttype, 'Resolution', resolution, 'Colorspace', colorspace)
bapf.set('VIEW', BrainSurfacePF.VIEW_AD_AZEL)
braph2print(gui_bapf.get('PF').get('H'), [fileparts(fileparts(which('braph2'))) filesep 'NetworkModelling' filesep 'atlas_AD.jpg'], 'ContentType', contenttype, 'Resolution', resolution, 'Colorspace', colorspace)
bapf.set('VIEW', BrainSurfacePF.VIEW_CA_AZEL)
braph2print(gui_bapf.get('PF').get('H'), [fileparts(fileparts(which('braph2'))) filesep 'NetworkModelling' filesep 'atlas_CA.jpg'], 'ContentType', contenttype, 'Resolution', resolution, 'Colorspace', colorspace)

%% Load structural data
im_gr = ImporterGroupSubjectCON_XLS( ...
    'DIRECTORY', [fileparts(fileparts(which('braph2'))) filesep 'NetworkModelling' filesep 'data' filesep 'DTI'], ...
    'BA', ba, ...
    'WAITBAR', true ...
    );

gr = im_gr.get('GR');
g_dict = AnalyzeEnsemble_CON_WU('GR', gr).get('G_DICT');

%% Visualize structural Networks
gpf = GraphAdjPF('G', g_dict.get('IT', 50)); % change the idx to switch to other structural connectome
gui_g = GUIFig('PF', gpf);
gui_g.get('DRAW');
gui_g.get('SHOW')
braph2print(gui_g.get('PF').get('H'), [fileparts(fileparts(which('braph2'))) filesep 'NetworkModelling' filesep 'individual_structural_network.jpg'], 'ContentType', contenttype, 'Resolution', resolution, 'Colorspace', colorspace)

%% Load functional data
im_gr = ImporterGroupSubjectFUN_XLS( ...
    'DIRECTORY', [fileparts(fileparts(which('braph2'))) filesep 'NetworkModelling' filesep 'data' filesep 'fMRI'], ...
    'BA', ba, ...
    'WAITBAR', true ...
    );

gr = im_gr.get('GR');
g_dict = AnalyzeEnsemble_FUN_WU('GR', gr).get('G_DICT');

%% Visualize Functional Networks
gpf = GraphAdjPF('G', g_dict.get('IT', 1)); % change the idx to switch to other structural connectome
gui_g = GUIFig('PF', gpf);
gui_g.get('DRAW');
gui_g.get('SHOW')
braph2print(gui_g.get('PF').get('H'), [fileparts(fileparts(which('braph2'))) filesep 'NetworkModelling' filesep 'individual_functional_network.jpg'], 'ContentType', contenttype, 'Resolution', resolution, 'Colorspace', colorspace)

