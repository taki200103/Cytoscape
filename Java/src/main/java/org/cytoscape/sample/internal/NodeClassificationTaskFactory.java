package org.cytoscape.sample.internal;

import org.cytoscape.application.CyApplicationManager;
import org.cytoscape.view.model.CyNetworkViewManager;
import org.cytoscape.view.vizmap.VisualMappingManager;
import org.cytoscape.work.TaskFactory;
import org.cytoscape.work.TaskIterator;
import org.cytoscape.view.vizmap.VisualMappingFunctionFactory;

public class NodeClassificationTaskFactory implements TaskFactory {
    private final CyApplicationManager applicationManager;
    private final CyNetworkViewManager networkViewManager;
    private final VisualMappingManager visualMappingManager;
    private final VisualMappingFunctionFactory discreteMappingFactory;

    public NodeClassificationTaskFactory(CyApplicationManager applicationManager, CyNetworkViewManager networkViewManager, VisualMappingManager visualMappingManager, VisualMappingFunctionFactory discreteMappingFactory) {
        this.applicationManager = applicationManager;
        this.networkViewManager = networkViewManager;
        this.visualMappingManager = visualMappingManager;
        this.discreteMappingFactory = discreteMappingFactory;
    }

    @Override
    public TaskIterator createTaskIterator() {
        return new TaskIterator(new NodeClassificationTask(applicationManager, networkViewManager, visualMappingManager, discreteMappingFactory));
    }

    @Override
    public boolean isReady() {
        return true;
    }
}