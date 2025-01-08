package org.cytoscape.sample.internal;

import org.cytoscape.application.CyApplicationManager;
import org.cytoscape.view.model.CyNetworkViewManager;
import org.cytoscape.work.TaskFactory;
import org.cytoscape.work.TaskIterator;

public class GroupNodesTaskFactory implements TaskFactory {
    private final CyApplicationManager applicationManager;
    private final CyNetworkViewManager networkViewManager;

    public GroupNodesTaskFactory(CyApplicationManager applicationManager, CyNetworkViewManager networkViewManager) {
        this.applicationManager = applicationManager;
        this.networkViewManager = networkViewManager;
    }

    @Override
    public TaskIterator createTaskIterator() {
        return new TaskIterator(new GroupNodesTask(applicationManager, networkViewManager));
    }

    @Override
    public boolean isReady() {
        return applicationManager.getCurrentNetwork() != null;
    }
} 