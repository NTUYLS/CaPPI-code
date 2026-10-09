"""Manuscript-width vector protocol figure: 6.25 inches, labels >= 9 points."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
C = dict(ink='#233247', muted='#526176', line='#60758B', blue='#2E648C',
         blue_fill='#EDF4F9', teal='#27776F', teal_fill='#ECF6F3',
         amber='#A76B18', amber_fill='#FFF6E7', neutral='#F5F7FA')


def box(ax, x, y, width, height, text, style='blue', size=9.5, bold=False):
    edge = C.get(style, C['line'])
    face = C.get(style + '_fill', C['neutral'])
    ax.add_patch(FancyBboxPatch((x,y),width,height,
        boxstyle='round,pad=0.012,rounding_size=0.065', linewidth=.95,
        edgecolor=edge,facecolor=face,zorder=3))
    ax.text(x+width/2,y+height/2,text,ha='center',va='center',fontsize=size,
        color=C['ink'],weight='bold' if bold else 'normal',linespacing=1.24,zorder=4)


def arrow(ax, points, color='line', dashed=False):
    path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=9,
        color=C[color],linewidth=1.0,linestyle=(0,(3,2)) if dashed else '-',
        capstyle='round',joinstyle='round',zorder=2))


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    # 6.25 x 6.0 in becomes 445 x 427.2 pt in the manuscript. Retain labels
    # of at least 9.2 pt in the source (>=9.09 pt at manuscript width), with
    # explicitly rerouted compact lanes.
    fig=plt.figure(figsize=(6.25,6.00))
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,6.25);ax.set_ylim(0,6.00)
    ax.set_aspect('equal');ax.axis('off')
    ax.text(.25,5.76,'CaPPI evaluation protocol',ha='left',va='center',
            fontsize=12,weight='bold',color=C['ink'])
    box(ax,.70,5.16,4.85,.42,'Protein sequences + observed PPI pairs\nPutative negatives',
        style='neutral',size=10)
    box(ax,1.65,4.39,2.95,.50,'Split first\nPair / protein-disjoint folds',size=10,bold=True)
    arrow(ax,[(3.125,5.14),(3.125,4.91)])
    for x,label,style in [(.25,'Training pairs','blue'),(2.35,'Validation pairs','amber'),
                          (4.45,'Test pairs','teal')]:
        box(ax,x,3.79,1.50,.38,label,style=style,size=9.5,bold=True)
        arrow(ax,[(3.125,4.37),(3.125,4.26),(x+.75,4.26),(x+.75,4.19)])
    # Parallel sequence and graph branches: training positives alone form topology.
    box(ax,.25,3.03,1.50,.50,'Sequence encoding\nGCNN (trained)',size=9.3)
    box(ax,2.20,3.03,1.65,.50,'Train-positive graph\nNo val / test edges',size=9.2)
    box(ax,2.20,2.28,1.65,.46,'DRW + SGNS\n64D topology',size=10)
    box(ax,.25,1.55,3.60,.42,'Residual cross-modal fusion',size=10.5)
    box(ax,.25,.80,3.60,.50,'Symmetric MLP\nMean logits for (u,v) and (v,u)',size=10)
    arrow(ax,[(1.,3.77),(1.,3.55)])
    arrow(ax,[(1.,3.77),(1.,3.66),(3.025,3.66),(3.025,3.55)])
    arrow(ax,[(3.025,3.01),(3.025,2.76)])
    arrow(ax,[(1.,3.01),(1.,1.99)])
    arrow(ax,[(3.025,2.26),(3.025,1.99)])
    arrow(ax,[(2.05,1.53),(2.05,1.32)])
    # Validation selects the checkpoint; held-out test pairs never tune the model.
    box(ax,4.45,2.46,1.50,.54,'Validation BCE\nEarly stopping',style='amber',size=9.5)
    box(ax,4.45,1.56,1.50,.50,'Best validation\ncheckpoint',style='neutral',size=9.5)
    box(ax,4.45,.80,1.50,.50,'Final held-out test\nNo tuning',style='teal',size=9.2)
    arrow(ax,[(3.87,3.98),(4.21,3.98),(4.21,3.18),(5.20,3.18),(5.20,3.02)],color='amber')
    arrow(ax,[(5.20,2.44),(5.20,2.08)],color='amber')
    arrow(ax,[(5.20,1.54),(5.20,1.32)],color='teal')
    arrow(ax,[(5.97,3.98),(6.10,3.98),(6.10,1.05),(5.97,1.05)],color='teal')
    arrow(ax,[(3.87,1.05),(4.05,1.05),(4.05,2.73),(4.43,2.73)],dashed=True)
    ax.text(4.11,1.98,'model',rotation=90,ha='left',va='center',fontsize=9.2,
            color=C['muted'],zorder=5,bbox=dict(facecolor='white',edgecolor='none',pad=1.0))
    ax.text(.25,.44,'Training BCE updates GCNN, fusion and MLP.',ha='left',va='center',
            fontsize=9.5,color=C['muted'])
    ax.text(.25,.16,'Unseen / isolated protein: zero topology; sequence branch retained.',
            ha='left',va='center',fontsize=9.2,color=C['muted'])
    output=HERE/'results';output.mkdir(parents=True,exist_ok=True)
    for extension in ('png','pdf','svg'):
        path=output/('figure_workflow.'+extension);fig.savefig(path,dpi=300);print(str(path))
    plt.close(fig)


if __name__=='__main__':
    main()
