import { Text } from '@chakra-ui/react';
import PTCompanies from './PTCompanies';
import PTDefinition from './PTDefinition';
import PTFeatures from './PTFeatures';
import PTTrends from './PTTrends';

const PTSection = () => {
  return (
    <>
      {/* 6. PyTorch */}
      <Text as='b' fontSize='xl'>6. PyTorch</Text>
      <PTDefinition/>
      <PTFeatures/>
      <PTTrends/>
      <PTCompanies/>
    </>
  )
}

export default PTSection