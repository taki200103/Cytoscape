package org.cytoscape.sample.internal;

import org.cytoscape.work.AbstractTask;
import org.cytoscape.work.TaskMonitor;
import org.cytoscape.application.CyApplicationManager;
import org.cytoscape.model.CyNetwork;
import org.cytoscape.model.CyNode;
import org.cytoscape.view.model.CyNetworkView;
import org.cytoscape.view.model.CyNetworkViewManager;
import org.cytoscape.view.model.View;
import org.cytoscape.view.presentation.property.BasicVisualLexicon;

import java.util.*;

public class GroupNodesTask extends AbstractTask {
    private final CyApplicationManager applicationManager;
    private final CyNetworkViewManager networkViewManager;

    public GroupNodesTask(CyApplicationManager applicationManager, CyNetworkViewManager networkViewManager) {
        this.applicationManager = applicationManager;
        this.networkViewManager = networkViewManager;
    }

    @Override
    public void run(TaskMonitor taskMonitor) {
        taskMonitor.setTitle("Group Nodes by Label");
        
        CyNetwork network = applicationManager.getCurrentNetwork();
        if (network == null) {
            return;
        }

        Collection<CyNetworkView> views = networkViewManager.getNetworkViews(network);
        if (views.isEmpty()) {
            return;
        }
        CyNetworkView networkView = views.iterator().next();

        Map<Integer, List<CyNode>> labelGroups = new HashMap<>();
        
        // Group nodes by label
        for (CyNode node : network.getNodeList()) {
            Integer label = network.getRow(node).get("label", Integer.class);
            if (label != null) {
                if (!labelGroups.containsKey(label)) {
                    labelGroups.put(label, new ArrayList<>());
                }
                labelGroups.get(label).add(node);
            }
        }

        // Position nodes in groups
        int groupX = 0;
        int groupY = 0;
        int groupSpacing = 200;
        int nodeSpacing = 80;
        
        for (Map.Entry<Integer, List<CyNode>> entry : labelGroups.entrySet()) {
            List<CyNode> nodes = entry.getValue();
            int nodesPerRow = (int) Math.ceil(Math.sqrt(nodes.size()));
            
            for (int i = 0; i < nodes.size(); i++) {
                View<CyNode> nodeView = networkView.getNodeView(nodes.get(i));
                
                // Calculate position within group
                int row = i / nodesPerRow;
                int col = i % nodesPerRow;
                
                double x = groupX + (col * nodeSpacing);
                double y = groupY + (row * nodeSpacing);
                
                nodeView.setVisualProperty(BasicVisualLexicon.NODE_X_LOCATION, x);
                nodeView.setVisualProperty(BasicVisualLexicon.NODE_Y_LOCATION, y);
            }
            
            // Move to next group position
            groupX += groupSpacing + (nodesPerRow * nodeSpacing);
            if (groupX > 2000) {
                groupX = 0;
                groupY += groupSpacing + ((nodes.size() / nodesPerRow + 1) * nodeSpacing);
            }
        }
        
        networkView.updateView();
    }
} 